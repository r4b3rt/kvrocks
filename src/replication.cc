#include <arpa/inet.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <glog/logging.h>
#include <netinet/tcp.h>
#include <future>
#include <string>
#include <thread>

#include "redis_reply.h"
#include "replication.h"
#include "rocksdb_crc32c.h"
#include "sock_util.h"
#include "status.h"

void send_string(bufferevent *bev, const std::string &data) {
  auto output = bufferevent_get_output(bev);
  evbuffer_add(output, data.c_str(), data.length());
}

void ReplicationThread::CallbacksStateMachine::ConnEventCB(
    bufferevent *bev, short events, void *state_machine_ptr) {
  if (events & BEV_EVENT_CONNECTED) {
    // call write_cb when connected
    bufferevent_data_cb write_cb;
    bufferevent_getcb(bev, nullptr, &write_cb, nullptr, nullptr);
    write_cb(bev, state_machine_ptr);
    return;
  }
  if (events & (BEV_EVENT_ERROR | BEV_EVENT_EOF)) {
    LOG(ERROR) << "[replication] connection error/eof, reconnect";
    // Wait a bit and reconnect
    auto state_m = static_cast<CallbacksStateMachine *>(state_machine_ptr);
    state_m->repl_->repl_state_ = kReplConnecting;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    state_m->Stop();
    state_m->Start();
  }
}

void ReplicationThread::CallbacksStateMachine::SetReadCB(
    bufferevent *bev, bufferevent_data_cb cb, void *state_machine_ptr) {
  bufferevent_enable(bev, EV_READ);
  bufferevent_setcb(bev, cb, nullptr, ConnEventCB, state_machine_ptr);
}

void ReplicationThread::CallbacksStateMachine::SetWriteCB(
    bufferevent *bev, bufferevent_data_cb cb, void *state_machine_ptr) {
  bufferevent_enable(bev, EV_WRITE);
  bufferevent_setcb(bev, nullptr, cb, ConnEventCB, state_machine_ptr);
}

ReplicationThread::CallbacksStateMachine::CallbacksStateMachine(
    ReplicationThread *repl,
    ReplicationThread::CallbacksStateMachine::CallbackList &&handlers)
    : repl_(repl), handlers_(std::move(handlers)) {
  if (!repl_->auth_.empty()) {
    handlers_.emplace_front(CallbacksStateMachine::READ, "auth read", authReadCB);
    handlers_.emplace_front(CallbacksStateMachine::WRITE, "auth write", authWriteCB);
  }
}

void ReplicationThread::CallbacksStateMachine::EvCallback(bufferevent *bev,
                                                          void *ctx) {
  auto self = static_cast<CallbacksStateMachine *>(ctx);
LOOP_LABEL:
  assert(self->handler_idx_ <= self->handlers_.size());
  auto st = self->getHandlerFunc(self->handler_idx_)(bev, self->repl_);
  DLOG(INFO) << "[replication] Execute handler[" << self->getHandlerName(self->handler_idx_) << "]";
  switch (st) {
    case CBState::NEXT:
      ++self->handler_idx_;
      if (self->getHandlerEventType(self->handler_idx_) == WRITE) {
        SetWriteCB(bev, EvCallback, ctx);
      } else {
        SetReadCB(bev, EvCallback, ctx);
      }
      // invoke the read handler (of next step) directly, as the bev might
      // have the data already.
      goto LOOP_LABEL;
    case CBState::AGAIN:
      break;
    case CBState::QUIT: // state that can not be retry, or all steps are executed.
      bufferevent_free(bev);
      self->bev_ = nullptr;
      self->repl_->repl_state_ = kReplError;
      break;
    case CBState::RESTART: // state that can be retried some time later
      self->Stop();
      LOG(INFO) << "[replication] Retry in 10 seconds";
      std::this_thread::sleep_for(std::chrono::seconds(10));
      self->Start();
  }
}

void ReplicationThread::CallbacksStateMachine::Start() {
  if (handlers_.empty()) {
    return;
  }
  auto sockaddr_inet = NewSockaddrInet(repl_->host_, repl_->port_);
  auto bev = bufferevent_socket_new(repl_->base_, -1, BEV_OPT_CLOSE_ON_FREE);
  if (bufferevent_socket_connect(bev,
                                 reinterpret_cast<sockaddr *>(&sockaddr_inet),
                                 sizeof(sockaddr_inet)) != 0) {
    // NOTE: Connection error will not appear here, network err will be reported
    // in ConnEventCB. the error here is something fatal.
    LOG(ERROR) << "[replication] Failed to start";
  }
  handler_idx_ = 0;
  if (getHandlerEventType(0) == WRITE) {
    SetWriteCB(bev, EvCallback, this);
  } else {
    SetReadCB(bev, EvCallback, this);
  }
  bev_ = bev;
}

void ReplicationThread::CallbacksStateMachine::Stop() {
  if (bev_) {
    bufferevent_free(bev_);
    bev_ = nullptr;
  }
}

ReplicationThread::ReplicationThread(std::string host, uint32_t port,
                                     Engine::Storage *storage, std::string auth)
    : host_(std::move(host)),
      port_(port),
      auth_(std::move(auth)),
      storage_(storage),
      repl_state_(kReplConnecting),
      psync_steps_(this,
                   CallbacksStateMachine::CallbackList{
                       CallbacksStateMachine::CallbackType{
                           CallbacksStateMachine::WRITE, "dbname write",checkDBNameWriteCB
                       },
                       CallbacksStateMachine::CallbackType{
                           CallbacksStateMachine::READ, "dbname read", checkDBNameReadCB
                       },
                       CallbacksStateMachine::CallbackType{
                           CallbacksStateMachine::WRITE, "psync write", tryPSyncWriteCB
                       },
                       CallbacksStateMachine::CallbackType{
                           CallbacksStateMachine::READ, "psync read", tryPSyncReadCB
                       },
                       CallbacksStateMachine::CallbackType{
                           CallbacksStateMachine::READ, "batch loop", incrementBatchLoopCB
                       }
                   }),
      fullsync_steps_(this,
                      CallbacksStateMachine::CallbackList{
                          CallbacksStateMachine::CallbackType{
                              CallbacksStateMachine::WRITE, "fullsync write",fullSyncWriteCB
                          },
                          CallbacksStateMachine::CallbackType{
                              CallbacksStateMachine::READ, "fullsync read", fullSyncReadCB}
                      }) {
  seq_ = storage_->LatestSeq();
}

void ReplicationThread::Start(std::function<void()> &&pre_fullsync_cb,
                              std::function<void()> &&post_fullsync_cb) {
  pre_fullsync_cb_ = std::move(pre_fullsync_cb);
  post_fullsync_cb_ = std::move(post_fullsync_cb);

  // Remove the backup_dir, so we can start replication in a clean state
  if (!Engine::Storage::BackupManager::PurgeBackup(storage_).IsOK()) {
    LOG(ERROR) << "[replication] Failed to purge existed backup dir";
  }

  try {
    t_ = std::thread([this]() {
      this->run();
      assert(stop_flag_);
    });
  } catch (const std::system_error &e) {
    LOG(ERROR) << "[replication] Failed to create thread: " << e.what();
    return;
  }
  LOG(INFO) << "[replication] Start";
}

void ReplicationThread::Stop() {
  stop_flag_ = true;  // Stopping procedure is asynchronous,
                      // handled by timer
  t_.join();
  LOG(INFO) << "[replication] Stopped";
}

/*
 * Run connect to master, and start the following steps
 * asynchronously
 *  - CheckDBName
 *  - TryPsync
 *  - - if ok, IncrementBatchLoop
 *  - - not, FullSync and restart TryPsync when done
 */
void ReplicationThread::run() {
  base_ = event_base_new();
  if (base_ == nullptr) {
    LOG(ERROR) << "[replication] Failed to create new ev base";
    return;
  }
  psync_steps_.Start();

  auto timer = event_new(base_, -1, EV_PERSIST, EventTimerCB, this);
  timeval tmo{1, 0};  // 1 sec
  evtimer_add(timer, &tmo);

  event_base_dispatch(base_);
  event_base_free(base_);
}

ReplicationThread::CBState ReplicationThread::authWriteCB(bufferevent *bev,
                                                          void *ctx) {
  auto self = static_cast<ReplicationThread *>(ctx);
  const auto auth_len_str = std::to_string(self->auth_.length());
  send_string(bev, "*2" CRLF "$4" CRLF "auth" CRLF "$" + auth_len_str + CRLF +
                       self->auth_ + CRLF);
  self->repl_state_ = kReplSendAuth;
  return CBState::NEXT;
}

ReplicationThread::CBState ReplicationThread::authReadCB(bufferevent *bev,
                                                         void *ctx) {
  char *line;
  size_t line_len;
  auto input = bufferevent_get_input(bev);
  line = evbuffer_readln(input, &line_len, EVBUFFER_EOL_CRLF_STRICT);
  if (!line) return CBState::AGAIN;
  if (strncmp(line, "+OK", 3) != 0) {
    // Auth failed
    LOG(ERROR) << "[replication] Auth failed: " << line;
    return CBState::QUIT;
  }
  return CBState::NEXT;
}

ReplicationThread::CBState ReplicationThread::checkDBNameWriteCB(
    bufferevent *bev, void *ctx) {
  send_string(bev, "*1" CRLF "$8" CRLF "_db_name" CRLF);
  auto self = static_cast<ReplicationThread *>(ctx);
  self->repl_state_ = kReplCheckDBName;
  return CBState::NEXT;
}

ReplicationThread::CBState ReplicationThread::checkDBNameReadCB(
    bufferevent *bev, void *ctx) {
  char *line;
  size_t line_len;
  auto input = bufferevent_get_input(bev);
  line = evbuffer_readln(input, &line_len, EVBUFFER_EOL_CRLF_STRICT);
  if (!line) return CBState::AGAIN;

  auto self = static_cast<ReplicationThread *>(ctx);
  std::string db_name = self->storage_->GetName();
  if (line_len == db_name.size() && !strncmp(line, db_name.data(), line_len)) {
    // DB name match, we should continue to next step: TryPsync
    free(line);
    return CBState::NEXT;
  }
  free(line);
  LOG(ERROR) << "[replication] db-name mismatched: " << line;
  return CBState::QUIT;
}

ReplicationThread::CBState ReplicationThread::tryPSyncWriteCB(
    bufferevent *bev, void *ctx) {
  auto self = static_cast<ReplicationThread *>(ctx);
  const auto seq_str = std::to_string(self->seq_);
  const auto seq_len_str = std::to_string(seq_str.length());
  const auto cmd_str = "*2" CRLF "$5" CRLF "PSYNC" CRLF "$" + seq_len_str +
                       CRLF + seq_str + CRLF;
  send_string(bev, cmd_str);
  self->repl_state_ = kReplSendPSync;
  LOG(INFO) << "[replication] Try to use psync, from seq: " << self->seq_;
  return CBState::NEXT;
}

ReplicationThread::CBState ReplicationThread::tryPSyncReadCB(bufferevent *bev,
                                                             void *ctx) {
  char *line;
  size_t line_len;
  auto self = static_cast<ReplicationThread *>(ctx);
  auto input = bufferevent_get_input(bev);
  line = evbuffer_readln(input, &line_len, EVBUFFER_EOL_CRLF_STRICT);
  if (!line) return CBState::AGAIN;

  if (strncmp(line, "+OK", 3) != 0) {
    // PSYNC isn't OK, we should use FullSync
    // Switch to fullsync state machine
    self->fullsync_steps_.Start();
    LOG(INFO) << "[replication] Failed to use psync, switch to fullsync";
    return CBState::QUIT;
  } else {
    // PSYNC is OK, use IncrementBatchLoop
    LOG(INFO) << "[replication] PSync is ok, start increment batch loop";
    return CBState::NEXT;
  }
}

ReplicationThread::CBState ReplicationThread::incrementBatchLoopCB(
    bufferevent *bev, void *ctx) {
  char *line = nullptr;
  size_t line_len = 0;
  char *bulk_data = nullptr;
  auto self = static_cast<ReplicationThread *>(ctx);
  self->repl_state_ = kReplConnected;
  auto input = bufferevent_get_input(bev);
  while (true) {
    switch (self->incr_state_) {
      case Incr_batch_size:
        // Read bulk length
        line = evbuffer_readln(input, &line_len, EVBUFFER_EOL_CRLF_STRICT);
        if (!line) return CBState::AGAIN;
        self->incr_bulk_len_ = line_len > 0 ? std::strtoull(line + 1, nullptr, 10) : 0;
        free(line);
        if (self->incr_bulk_len_ == 0) {
          LOG(ERROR) << "[replication] Invalid increment data size";
          return CBState::RESTART;
        }
        self->incr_state_ = Incr_batch_data;
      case Incr_batch_data:
        // Read bulk data (batch data)
        if (self->incr_bulk_len_+2 <= evbuffer_get_length(input)) {  // We got enough data
          bulk_data = reinterpret_cast<char *>(evbuffer_pullup(input, self->incr_bulk_len_ + 2));
          auto s = self->storage_->WriteBatch(std::string(bulk_data, self->incr_bulk_len_));
          if (!s.IsOK()) {
            LOG(ERROR) << "[replication] CRITICAL - Failed to write batch to local, err: " << s.Msg();
            self->stop_flag_ = true; // This is a very critical error, data might be corrupted
            return CBState::QUIT;
          }
          evbuffer_drain(input, self->incr_bulk_len_ + 2);
          self->incr_state_ = Incr_batch_size;
        } else {
          return CBState::AGAIN;
        }
    }
  }
}

ReplicationThread::CBState ReplicationThread::fullSyncWriteCB(
    bufferevent *bev, void *ctx) {
  send_string(bev, "*1" CRLF "$11" CRLF "_fetch_meta" CRLF);
  auto self = static_cast<ReplicationThread *>(ctx);
  self->repl_state_ = kReplFetchMeta;
  return CBState::NEXT;
}

ReplicationThread::CBState ReplicationThread::fullSyncReadCB(bufferevent *bev,
                                                             void *ctx) {
  char *line;
  size_t line_len;
  auto self = static_cast<ReplicationThread *>(ctx);
  auto input = bufferevent_get_input(bev);
  switch (self->fullsync_state_) {
    case kFetchMetaID:
      line = evbuffer_readln(input, &line_len, EVBUFFER_EOL_CRLF_STRICT);
      if (!line) return CBState::AGAIN;
      if (line[0] == '-') {
        LOG(ERROR) << "[replication] Failed to fetch meta id: " << line;
        return CBState::RESTART;
      }
      self->fullsync_meta_id_ = static_cast<rocksdb::BackupID>(
          line_len > 0 ? std::strtoul(line, nullptr, 10) : 0);
      free(line);
      if (self->fullsync_meta_id_ == 0) {
        LOG(ERROR) << "[replication] Invalid meta id received";
        return CBState::RESTART;
      }
      self->fullsync_state_ = kFetchMetaSize;
      LOG(INFO) << "[replication] Success to fetch meta id: " << self->fullsync_meta_id_;
    case kFetchMetaSize:
      line = evbuffer_readln(input, &line_len, EVBUFFER_EOL_CRLF_STRICT);
      if (!line) return CBState::AGAIN;
      if (line[0] == '-') {
        LOG(ERROR) << "[replication] Failed to fetch meta size: " << line;
        return CBState::RESTART;
      }
      self->fullsync_filesize_ = line_len > 0 ? std::strtoull(line, nullptr, 10) : 0;
      free(line);
      if (self->fullsync_filesize_ == 0) {
        LOG(ERROR) << "[replication] Invalid meta file size received";
        return CBState::RESTART;
      }
      self->fullsync_state_ = kFetchMetaContent;
      LOG(INFO) << "[replication] Success to fetch meta size: " << self->fullsync_filesize_;
    case kFetchMetaContent:
      if (evbuffer_get_length(input) < self->fullsync_filesize_) {
        return CBState::AGAIN;
      }
      auto meta = Engine::Storage::BackupManager::ParseMetaAndSave(
          self->storage_, self->fullsync_meta_id_, input);
      assert(evbuffer_get_length(input) == 0);
      self->fullsync_state_ = kFetchMetaID;

      LOG(INFO) << "[replication] Succeeded fetching meta file, fetching files in parallel";
      self->repl_state_ = kReplFetchSST;
      if (!self->parallelFetchFile(meta.files).IsOK()) {
        return CBState::RESTART;
      }

      // Restore DB from backup
      self->pre_fullsync_cb_();
      if (!self->storage_->RestoreFromBackup(&self->seq_).IsOK()) {
        LOG(ERROR) << "[replication] Failed to restore backup";
        self->post_fullsync_cb_();
        return CBState::RESTART;
      }
      self->post_fullsync_cb_();
      ++self->seq_;

      // Switch to psync state machine again
      self->psync_steps_.Start();
      return CBState::QUIT;
  }

  LOG(ERROR) << "Should not arrive here";
  assert(false);
  return CBState::QUIT;
}

Status ReplicationThread::parallelFetchFile(const std::vector<std::pair<std::string, uint32_t>> &files) {
  size_t concurrency = 1;
  if (files.size() > 20) {
    // Use 4 threads to download files in parallel
    concurrency = 4;
  }
  std::vector<std::future<bool>> results;
  for (size_t tid = 0; tid < concurrency; ++tid) {
    results.push_back(std::async(
        std::launch::async, [this, &files, tid, concurrency]() -> bool {
          if (this->stop_flag_) {
            return false;
          }
          int sock_fd;
          if (SockConnect(this->host_, this->port_, &sock_fd) < 0) {
            LOG(ERROR) << "[replication] Failed to connect the repl port, err: " << strerror(errno);
            return false;
          }
          if (!this->sendAuth(sock_fd).IsOK()) {
            return false;
          }
          for (auto f_idx = tid; f_idx < files.size();
               f_idx += concurrency) {
            const auto &f_name = files[f_idx].first;
            const auto &f_crc = files[f_idx].second;
            // Don't fetch existing files
            if (Engine::Storage::BackupManager::FileExists(this->storage_,
                                                           f_name)) {
              LOG(INFO) << "[skip] "<< f_name << " " << f_crc;
              continue;
            }
            DLOG(INFO) << "[fetch] " << f_name << " " << f_crc;
            auto s = this->fetchFile(sock_fd, f_name, f_crc);
            if (!s.IsOK()) {
              LOG(ERROR) << "[replication] Failed to fetch sst file, err: " << s.Msg();
              return false;
            }
          }
          close(sock_fd);
          return true;
        }));
  }

  // Wait til finish
  for (auto &f : results) {
    if (!f.get()) {
      LOG(ERROR) << "[replication] Failed to fetch some files";
      return Status(Status::NotOK);
    }
  }
  return Status::OK();
}

Status ReplicationThread::sendAuth(int sock_fd) {
  char *line;
  size_t line_len;
  evbuffer *evbuf = evbuffer_new();

  // Send auth when needed
  if (!auth_.empty()) {
    const auto auth_len_str = std::to_string(auth_.length());
    SockSend(sock_fd, "*2" CRLF "$4" CRLF "auth" CRLF "$" + auth_len_str +
        CRLF + auth_ + CRLF);
    while (true) {
      if (evbuffer_read(evbuf, sock_fd, -1) < 0) {
        LOG(ERROR) << "[replication] Failed to auth resp";
        return Status(Status::NotOK);
      }
      line = evbuffer_readln(evbuf, &line_len, EVBUFFER_EOL_CRLF_STRICT);
      if (!line) continue;
      if (strncmp(line, "+OK", 3) != 0) {
        LOG(ERROR) << "[replication] Auth failed";
        return Status(Status::NotOK);
      }
      break;
    }
  }
  return Status::OK();
}


Status ReplicationThread::fetchFile(int sock_fd, std::string path,
                                    uint32_t crc) {
  char *line;
  size_t line_len, file_size;
  evbuffer *evbuf = evbuffer_new();

  const auto cmd_str = "*2" CRLF "$11" CRLF "_fetch_file" CRLF "$" +
                       std::to_string(path.length()) + CRLF + path + CRLF;
  if (SockSend(sock_fd, cmd_str) < 0) {
    return Status(Status::NotOK);
  }

  // Read file size line
  while (true) {
    if (evbuffer_read(evbuf, sock_fd, -1) < 0) {
      return Status(Status::NotOK, std::string("read size line err: ")+strerror(errno));
    }
    line = evbuffer_readln(evbuf, &line_len, EVBUFFER_EOL_CRLF_STRICT);
    if (!line) continue;
    if (*line == '-') {
      return Status(Status::NotOK, std::string("_fetch_file got err: ")+line);
    }
    file_size = line_len > 0 ? std::strtoull(line, nullptr, 10) : 0;
    free(line);
    break;
  }
  // Write to tmp file
  auto tmp_file = Engine::Storage::BackupManager::NewTmpFile(storage_, path);
  if (!tmp_file) {
    return Status(Status::NotOK, "unable to create tmp file");
  }

  size_t seen_bytes = 0;
  uint32_t tmp_crc = 0;
  char data[1024];
  while (seen_bytes < file_size) {
    if (evbuffer_get_length(evbuf) > 0) {
      auto data_len = evbuffer_remove(evbuf, data, 1024);
      if (data_len == 0) continue;
      if (data_len < 0) {
        return Status(Status::NotOK, "read sst file data error");
      }
      tmp_file->Append(rocksdb::Slice(data, data_len));
      tmp_crc = rocksdb::crc32c::Extend(tmp_crc, data, data_len);
      seen_bytes += data_len;
    } else {
      if (evbuffer_read(evbuf, sock_fd, -1) < 0) {
        return Status(Status::NotOK, std::string("read sst file data, err: ")+strerror(errno));
      }
    }
  }
  if (crc != tmp_crc) {
    return Status(Status::NotOK, "CRC mismatch");
  }
  // File is OK, rename to formal name
  return Engine::Storage::BackupManager::SwapTmpFile(storage_, path);
}

// Check if stop_flag_ is set, when do, tear down replication
void ReplicationThread::EventTimerCB(int, short, void *ctx) {
  // DLOG(INFO) << "[replication] timer";
  auto self = static_cast<ReplicationThread *>(ctx);
  if (self->stop_flag_) {
    LOG(INFO) << "[replication] Stop ev loop";
    event_base_loopbreak(self->base_);
    self->psync_steps_.Stop();
    self->fullsync_steps_.Stop();
  }
}
