[[clang::noinline]]
static void setup(kitsune::mobile_ptr<int> &dst, kitsune::mobile_ptr<int> &src,
                  long n) {
  srand(7);
  dst.alloc(n);
  src.alloc(n);

  for (long i = 0; i < n; ++i) {
    src[i] = rand();
    dst[i] = 0;
  }
}

[[clang::noinline]]
static void teardown(kitsune::mobile_ptr<int> &dst,
                     kitsune::mobile_ptr<int> &src) {
  dst.free();
  src.free();
}

[[clang::noinline]]
static long check(const kitsune::mobile_ptr<int> &dst,
                  const kitsune::mobile_ptr<int> &src, long n) {
  for (long i = 0; i < n; ++i)
    if (dst[i] != src[i])
      return i + 1;
  return 0;
}

[[clang::noinline]]
static int report(long err) {
  if (err)
    printf("ERROR at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
