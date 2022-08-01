#include <cstdio>

int main(void)
{
	int N, res = 0;
	scanf("%d", &N);
	
	for (int i=1; i<=N; i++) {
		res += i;
	}
	printf("%d\n", res);
	
	return 0;
}
