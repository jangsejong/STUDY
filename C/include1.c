#include <cstdio>

int main(void)
{
	int N, res;
	scanf("%d", &N);
	
	res = N * (N+1) / 2;
	printf("%d\n", res);
	
	return 0;
}
