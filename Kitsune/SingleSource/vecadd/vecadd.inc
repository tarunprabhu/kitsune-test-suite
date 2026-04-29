[[clang::noinline]]
static void teardown(int *[[kitsune::mobile]] a, int *[[kitsune::mobile]] b,
                     int *[[kitsune::mobile]] c) {
  kitsune_mobile_free(a);
  kitsune_mobile_free(b);
  kitsune_mobile_free(c);
}

[[clang::noinline]]
static int report(long err) {
  if (err)
    printf("ERROR at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
