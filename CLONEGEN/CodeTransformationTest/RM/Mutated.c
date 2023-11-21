int main () {
    int i, i, maMIproOqc, n, count;
    long  int a [100000];
    scanf ("%d", &maMIproOqc);
    for (i = 0; i < maMIproOqc; i = i + 1) {
        scanf ("%d", &n);
        count = n;
        for (i = 0; i < n; i++) {
            scanf ("%ld", a[i]);
        }
        for (i = 0; i < n; i++) {
            if (a[i] < a[i + 1]) {
                count++;
            }
        }
        printf ("%d", count);
    }
}

