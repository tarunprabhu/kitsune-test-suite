#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <kitsune.h>

static FILE *saved = NULL;

[[clang::noinline]]
static char *setup(size_t bufsz) {
  char *buf = (char *)calloc(bufsz, sizeof(char));
  saved = stdout;
  stdout = fmemopen(buf, bufsz, "w");

  return buf;
}

[[clang::noinline]]
static void teardown(char *buf) {
  fclose(stdout);
  stdout = saved;
  free(buf);
}

static long toDigit(char c) { return long(c) - long('0'); }

[[clang::noinline]]
static bool check(char *buf, const std::vector<std::vector<int>> &expected) {
  std::vector<int> tup;
  std::vector<std::vector<int>> tups;
  long num = 0;
  long i = 0;
  for (i = 0; buf[i] != '\0'; ++i) {
    if (buf[i] == ' ') {
      tup.push_back(num);
      num = 0;
    } else if (buf[i] == '\n') {
      tup.push_back(num);
      tups.push_back(tup);
      tup.clear();
      num = 0;
    } else {
      num = num * 10 + toDigit(buf[i]);
    }
  }

  std::sort(tups.begin(), tups.end());
  return tups != expected;
}

[[clang::noinline]]
static int report(bool err) {
  if (err)
    printf("FAIL\n");
  else
    printf("PASS\n");
  return err;
}
