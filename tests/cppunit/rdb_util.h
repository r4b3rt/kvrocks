/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */
#pragma once

#include <gtest/gtest.h>

#include <fstream>
#include <string>

// RDB test data, copy from Redis's tests/asset/*rdb, not shellcode.

// zset-ziplist.rdb
inline constexpr const char zset_ziplist_payload[] =
    "\x52\x45\x44\x49\x53\x30\x30\x31\x30\xfa\x09\x72\x65\x64\x69\x73\x2d\x76\x65\x72\x0b\x32\x35\x35\x2e"
    "\x32\x35\x35\x2e\x32\x35\x35\xfa\x0a\x72\x65\x64\x69\x73\x2d\x62\x69\x74\x73\xc0\x40\xfa\x05\x63\x74"
    "\x69\x6d\x65\xc2\x62\xb7\x13\x61\xfa\x08\x75\x73\x65\x64\x2d\x6d\x65\x6d\xc2\x50\xf4\x0c\x00\xfa\x0c"
    "\x61\x6f\x66\x2d\x70\x72\x65\x61\x6d\x62\x6c\x65\xc0\x00\xfe\x00\xfb\x01\x00\x0c\x04\x7a\x73\x65\x74"
    "\x19\x19\x00\x00\x00\x16\x00\x00\x00\x04\x00\x00\x03\x6f\x6e\x65\x05\xf2\x02\x03\x74\x77\x6f\x05\xf3"
    "\xff\xff\x1f\xb2\xfd\xf0\x99\x7f\x9e\x19";

// corrupt_empty_keys.rdb
inline constexpr const char corrupt_empty_keys_payload[] =
    "\x52\x45\x44\x49\x53\x30\x30\x31\x30\xfa\x09\x72\x65\x64\x69\x73\x2d\x76\x65\x72\x0b\x32\x35\x35\x2e"
    "\x32\x35\x35\x2e\x32\x35\x35\xfa\x0a\x72\x65\x64\x69\x73\x2d\x62\x69\x74\x73\xc0\x40\xfa\x05\x63\x74"
    "\x69\x6d\x65\xc2\x7a\x18\x15\x61\xfa\x08\x75\x73\x65\x64\x2d\x6d\x65\x6d\xc2\x80\x31\x10\x00\xfa\x0c"
    "\x61\x6f\x66\x2d\x70\x72\x65\x61\x6d\x62\x6c\x65\xc0\x00\xfe\x00\xfb\x09\x00\x02\x03\x73\x65\x74\x00"
    "\x04\x04\x68\x61\x73\x68\x00\x0a\x0c\x6c\x69\x73\x74\x5f\x7a\x69\x70\x6c\x69\x73\x74\x0b\x0b\x00\x00"
    "\x00\x0a\x00\x00\x00\x00\x00\xff\x05\x04\x7a\x73\x65\x74\x00\x11\x0d\x7a\x73\x65\x74\x5f\x6c\x69\x73"
    "\x74\x70\x61\x63\x6b\x07\x07\x00\x00\x00\x00\x00\xff\x10\x0c\x68\x61\x73\x68\x5f\x7a\x69\x70\x6c\x69"
    "\x73\x74\x07\x07\x00\x00\x00\x00\x00\xff\x0e\x0e\x6c\x69\x73\x74\x5f\x71\x75\x69\x63\x6b\x6c\x69\x73"
    "\x74\x00\x0c\x0c\x7a\x73\x65\x74\x5f\x7a\x69\x70\x6c\x69\x73\x74\x0b\x0b\x00\x00\x00\x0a\x00\x00\x00"
    "\x00\x00\xff\x0e\x1c\x6c\x69\x73\x74\x5f\x71\x75\x69\x63\x6b\x6c\x69\x73\x74\x5f\x65\x6d\x70\x74\x79"
    "\x5f\x7a\x69\x70\x6c\x69\x73\x74\x01\x0b\x0b\x00\x00\x00\x0a\x00\x00\x00\x00\x00\xff\xff\xf0\xf5\x06"
    "\xdd\xc6\x6e\x61\x83";

// encodings.rdb
inline constexpr const char encodings_rdb_payload[] =
    "\x52\x45\x44\x49\x53\x30\x30\x30\x34\xfe\x00\x03\x04\x7a\x73\x65\x74\x0c\x02\x62\x62\x02\x32\x30\x02\x63\x63\x02"
    "\x33\x30"
    "\x03\x62\x62\x62\x03\x32\x30\x30\x04\x62\x62\x62\x62\x0a\x35\x30\x30\x30\x30\x30\x30\x30\x30\x30\x03\x63\x63\x63"
    "\x03\x33"
    "\x30\x30\x04\x63\x63\x63\x63\x09\x31\x32\x33\x34\x35\x36\x37\x38\x39\x01\x61\x01\x31\x02\x61\x61\x02\x31\x30\x01"
    "\x62\x01"
    "\x32\x03\x61\x61\x61\x03\x31\x30\x30\x01\x63\x01\x33\x04\x61\x61\x61\x61\x04\x31\x30\x30\x30\x0b\x0c\x73\x65\x74"
    "\x5f\x7a"
    "\x69\x70\x70\x65\x64\x5f\x31\x10\x02\x00\x00\x00\x04\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x0a\x0b\x6c\x69"
    "\x73\x74"
    "\x5f\x7a\x69\x70\x70\x65\x64\x30\x30\x00\x00\x00\x25\x00\x00\x00\x08\x00\x00\xc0\x01\x00\x04\xc0\x02\x00\x04\xc0"
    "\x03\x00"
    "\x04\x01\x61\x03\x01\x62\x03\x01\x63\x03\xd0\xa0\x86\x01\x00\x06\xe0\x00\xbc\xa0\x65\x01\x00\x00\x00\xff\x00\x06"
    "\x73\x74"
    "\x72\x69\x6e\x67\x0b\x48\x65\x6c\x6c\x6f\x20\x57\x6f\x72\x6c\x64\x00\x0c\x63\x6f\x6d\x70\x72\x65\x73\x73\x69\x62"
    "\x6c\x65"
    "\xc3\x09\x40\x89\x01\x61\x61\xe0\x7c\x00\x01\x61\x61\x0b\x0c\x73\x65\x74\x5f\x7a\x69\x70\x70\x65\x64\x5f\x32\x18"
    "\x04\x00"
    "\x00\x00\x04\x00\x00\x00\xa0\x86\x01\x00\x40\x0d\x03\x00\xe0\x93\x04\x00\x80\x1a\x06\x00\x00\x06\x6e\x75\x6d\x62"
    "\x65\x72"
    "\xc0\x0a\x0b\x0c\x73\x65\x74\x5f\x7a\x69\x70\x70\x65\x64\x5f\x33\x38\x08\x00\x00\x00\x06\x00\x00\x00\x00\xca\x9a"
    "\x3b\x00"
    "\x00\x00\x00\x00\x94\x35\x77\x00\x00\x00\x00\x00\x5e\xd0\xb2\x00\x00\x00\x00\x00\x28\x6b\xee\x00\x00\x00\x00\x00"
    "\xf2\x05"
    "\x2a\x01\x00\x00\x00\x00\xbc\xa0\x65\x01\x00\x00\x00\x02\x03\x73\x65\x74\x08\x0a\x36\x30\x30\x30\x30\x30\x30\x30"
    "\x30\x30"
    "\xc2\xa0\x86\x01\x00\x01\x61\xc0\x01\x01\x62\xc0\x02\x01\x63\xc0\x03\x01\x04\x6c\x69\x73\x74\x18\xc0\x01\xc0\x02"
    "\xc0\x03"
    "\x01\x61\x01\x62\x01\x63\xc2\xa0\x86\x01\x00\x0a\x36\x30\x30\x30\x30\x30\x30\x30\x30\x30\xc0\x01\xc0\x02\xc0\x03"
    "\x01\x61"
    "\x01\x62\x01\x63\xc2\xa0\x86\x01\x00\x0a\x36\x30\x30\x30\x30\x30\x30\x30\x30\x30\xc0\x01\xc0\x02\xc0\x03\x01\x61"
    "\x01\x62"
    "\x01\x63\xc2\xa0\x86\x01\x00\x0a\x36\x30\x30\x30\x30\x30\x30\x30\x30\x30\x0d\x0b\x68\x61\x73\x68\x5f\x7a\x69\x70"
    "\x70\x65"
    "\x64\x20\x20\x00\x00\x00\x1b\x00\x00\x00\x06\x00\x00\x01\x61\x03\xc0\x01\x00\x04\x01\x62\x03\xc0\x02\x00\x04\x01"
    "\x63\x03"
    "\xc0\x03\x00\xff\x0c\x0b\x7a\x73\x65\x74\x5f\x7a\x69\x70\x70\x65\x64\x20\x20\x00\x00\x00\x1b\x00\x00\x00\x06\x00"
    "\x00\x01"
    "\x61\x03\xc0\x01\x00\x04\x01\x62\x03\xc0\x02\x00\x04\x01\x63\x03\xc0\x03\x00\xff\x04\x04\x68\x61\x73\x68\x0b\x01"
    "\x62\xc0"
    "\x02\x02\x61\x61\xc0\x0a\x01\x63\xc0\x03\x03\x61\x61\x61\xc0\x64\x02\x62\x62\xc0\x14\x02\x63\x63\xc0\x1e\x03\x62"
    "\x62\x62"
    "\xc1\xc8\x00\x03\x63\x63\x63\xc1\x2c\x01\x03\x64\x64\x64\xc1\x90\x01\x03\x65\x65\x65\x0a\x35\x30\x30\x30\x30\x30"
    "\x30\x30"
    "\x30\x30\x01\x61\xc0\x01\xff";

// hash-ziplist.rdb
inline constexpr const char hash_ziplist_payload[] =
    "\x52\x45\x44\x49\x53\x30\x30\x30\x39\xfa\x09\x72\x65\x64\x69\x73\x2d\x76\x65\x72\x0b\x32\x35\x35\x2e\x32\x35\x35"
    "\x2e\x32"
    "\x35\x35\xfa\x0a\x72\x65\x64\x69\x73\x2d\x62\x69\x74\x73\xc0\x40\xfa\x05\x63\x74\x69\x6d\x65\xc2\xc8\x5c\x96\x60"
    "\xfa\x08"
    "\x75\x73\x65\x64\x2d\x6d\x65\x6d\xc2\x90\xad\x0c\x00\xfa\x0c\x61\x6f\x66\x2d\x70\x72\x65\x61\x6d\x62\x6c\x65\xc0"
    "\x00\xfe"
    "\x00\xfb\x01\x00\x0d\x04\x68\x61\x73\x68\x1b\x1b\x00\x00\x00\x16\x00\x00\x00\x04\x00\x00\x02\x66\x31\x04\x02\x76"
    "\x31\x04"
    "\x02\x66\x32\x04\x02\x76\x32\xff\xff\x4f\x9c\xd1\xfd\x16\x69\x98\x83";

// hash-zipmap.rdb
inline constexpr const char hash_zipmap_payload[] =
    "\x52\x45\x44\x49\x53\x30\x30\x30\x33\xfe\x00\x09\x04\x68\x61\x73\x68\x10\x02\x02\x66\x31\x02\x00\x76\x31\x02\x66"
    "\x32\x02"
    "\x00\x76\x32\xff\xff";

// list-quicklist.rdb
inline constexpr const char list_quicklist_payload[] =
    "\x52\x45\x44\x49\x53\x30\x30\x30\x38\xfa\x09\x72\x65\x64\x69\x73\x2d\x76\x65\x72\x05\x34\x2e\x30\x2e\x39\xfa\x0a"
    "\x72\x65"
    "\x64\x69\x73\x2d\x62\x69\x74\x73\xc0\x40\xfa\x05\x63\x74\x69\x6d\x65\xc2\x9f\x06\x26\x61\xfa\x08\x75\x73\x65\x64"
    "\x2d\x6d"
    "\x65\x6d\xc2\x80\x92\x07\x00\xfa\x0c\x61\x6f\x66\x2d\x70\x72\x65\x61\x6d\x62\x6c\x65\xc0\x00\xfe\x00\xfb\x02\x00"
    "\x0e\x04"
    "\x6c\x69\x73\x74\x01\x0d\x0d\x00\x00\x00\x0a\x00\x00\x00\x01\x00\x00\xf8\xff\x00\x01\x78\xc0\x07\xff\x35\x72\xf8"
    "\x54\x1a"
    "\xc4\xd7\x40";

// dumped from redis-server 7.0, sourced from the 'encodings.rdb' file.
inline constexpr const char encodings_ver10_rdb_payload[] =
    "\x52\x45\x44\x49\x53\x30\x30\x31\x30\xfa\x09\x72\x65\x64\x69\x73\x2d\x76\x65\x72\x05\x37\x2e\x30\x2e\x33\xfa\x0a"
    "\x72\x65"
    "\x64\x69\x73\x2d\x62\x69\x74\x73\xc0\x40\xfa\x05\x63\x74\x69\x6d\x65\xc2\x62\x65\x23\x65\xfa\x08\x75\x73\x65\x64"
    "\x2d\x6d"
    "\x65\x6d\xc2\x28\x4f\x0e\x00\xfa\x08\x61\x6f\x66\x2d\x62\x61\x73\x65\xc0\x00\xfe\x00\xfb\x0d\x00\x11\x04\x7a\x73"
    "\x65\x74"
    "\x40\x64\x64\x00\x00\x00\x18\x00\x81\x61\x02\x01\x01\x81\x62\x02\x02\x01\x81\x63\x02\x03\x01\x82\x61\x61\x03\x0a"
    "\x01\x82"
    "\x62\x62\x03\x14\x01\x82\x63\x63\x03\x1e\x01\x83\x61\x61\x61\x04\x64\x01\x83\x62\x62\x62\x04\xc0\xc8\x02\x83\x63"
    "\x63\x63"
    "\x04\xc1\x2c\x02\x84\x61\x61\x61\x61\x05\xc3\xe8\x02\x84\x63\x63\x63\x63\x05\xf3\x15\xcd\x5b\x07\x05\x84\x62\x62"
    "\x62\x62"
    "\x05\xf4\x00\xf2\x05\x2a\x01\x00\x00\x00\x09\xff\x11\x0b\x7a\x73\x65\x74\x5f\x7a\x69\x70\x70\x65\x64\x16\x16\x00"
    "\x00\x00"
    "\x06\x00\x81\x61\x02\x01\x01\x81\x62\x02\x02\x01\x81\x63\x02\x03\x01\xff\x10\x04\x68\x61\x73\x68\x40\x56\x56\x00"
    "\x00\x00"
    "\x16\x00\x81\x62\x02\x02\x01\x82\x61\x61\x03\x0a\x01\x81\x63\x02\x03\x01\x83\x61\x61\x61\x04\x64\x01\x82\x62\x62"
    "\x03\x14"
    "\x01\x82\x63\x63\x03\x1e\x01\x83\x62\x62\x62\x04\xc0\xc8\x02\x83\x63\x63\x63\x04\xc1\x2c\x02\x83\x64\x64\x64\x04"
    "\xc1\x90"
    "\x02\x83\x65\x65\x65\x04\xf4\x00\xf2\x05\x2a\x01\x00\x00\x00\x09\x81\x61\x02\x01\x01\xff\x0b\x0c\x73\x65\x74\x5f"
    "\x7a\x69"
    "\x70\x70\x65\x64\x5f\x32\x18\x04\x00\x00\x00\x04\x00\x00\x00\xa0\x86\x01\x00\x40\x0d\x03\x00\xe0\x93\x04\x00\x80"
    "\x1a\x06"
    "\x00\x12\x04\x6c\x69\x73\x74\x01\x02\xc3\x2b\x40\x61\x1f\x61\x00\x00\x00\x18\x00\x01\x01\x02\x01\x03\x01\x81\x61"
    "\x02\x81"
    "\x62\x02\x81\x63\x02\xf2\xa0\x86\x01\x04\xf4\x00\xbc\xa0\x65\x01\x20\x1e\x00\x09\xe0\x32\x1d\x01\x09\xff\x02\x03"
    "\x73\x65"
    "\x74\x08\xc0\x02\xc0\x01\xc2\xa0\x86\x01\x00\x01\x62\x0a\x36\x30\x30\x30\x30\x30\x30\x30\x30\x30\xc0\x03\x01\x61"
    "\x01\x63"
    "\x00\x06\x6e\x75\x6d\x62\x65\x72\xc0\x0a\x0b\x0c\x73\x65\x74\x5f\x7a\x69\x70\x70\x65\x64\x5f\x31\x10\x02\x00\x00"
    "\x00\x04"
    "\x00\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x10\x0b\x68\x61\x73\x68\x5f\x7a\x69\x70\x70\x65\x64\x16\x16\x00\x00"
    "\x00\x06"
    "\x00\x81\x61\x02\x01\x01\x81\x62\x02\x02\x01\x81\x63\x02\x03\x01\xff\x00\x0c\x63\x6f\x6d\x70\x72\x65\x73\x73\x69"
    "\x62\x6c"
    "\x65\xc3\x09\x40\x89\x01\x61\x61\xe0\x7c\x00\x01\x61\x61\x00\x06\x73\x74\x72\x69\x6e\x67\x0b\x48\x65\x6c\x6c\x6f"
    "\x20\x57"
    "\x6f\x72\x6c\x64\x0b\x0c\x73\x65\x74\x5f\x7a\x69\x70\x70\x65\x64\x5f\x33\x38\x08\x00\x00\x00\x06\x00\x00\x00\x00"
    "\xca\x9a"
    "\x3b\x00\x00\x00\x00\x00\x94\x35\x77\x00\x00\x00\x00\x00\x5e\xd0\xb2\x00\x00\x00\x00\x00\x28\x6b\xee\x00\x00\x00"
    "\x00\x00"
    "\xf2\x05\x2a\x01\x00\x00\x00\x00\xbc\xa0\x65\x01\x00\x00\x00\x12\x0b\x6c\x69\x73\x74\x5f\x7a\x69\x70\x70\x65\x64"
    "\x01\x02"
    "\x25\x25\x00\x00\x00\x08\x00\x01\x01\x02\x01\x03\x01\x81\x61\x02\x81\x62\x02\x81\x63\x02\xf2\xa0\x86\x01\x04\xf4"
    "\x00\xbc"
    "\xa0\x65\x01\x00\x00\x00\x09\xff\xff\x58\xe7\x62\x56\x52\x9b\xdf\x6c";

class ScopedTestRDBFile {
 public:
  ScopedTestRDBFile(const std::string &name, const char *data, size_t len) : name_(name) {
    std::ofstream out_file(name, std::ios::out | std::ios::binary);
    if (!out_file) {
      EXPECT_TRUE(false);
    }

    out_file.write(data, static_cast<std::streamsize>(len));
    if (!out_file) {
      EXPECT_TRUE(false);
    }
    out_file.close();
  }

  ScopedTestRDBFile(const ScopedTestRDBFile &) = delete;
  ScopedTestRDBFile &operator=(const ScopedTestRDBFile &) = delete;

  ~ScopedTestRDBFile() { std::remove(name_.c_str()); }

 private:
  std::string name_;
};
