# Linux共享内存

调用shmget函数获取或创建共享内存
```
int shmget(key_t key, size_t size, int shmflg);

//示例
struct st_pid
{
    int pid;
    char name[51];
}
int shmid;
shmid = shmget(0x5005, sizeof(struct st_pid), 0640|IPC_CREAT);
//示例结束

//查看共享内存命令
ipcs -m

//删除共享内存
ipcrm -m [shmid]
```



调用shmat函数把共享内存连接到当前进程的地址空间

```
void *shmat(int shmid, const void *shmaddr, int shmflg);//后面两个参数一般设为0
```
调用shmdt函数把共享内存从当前进程中分离

```
int shmdt(const void *shmaddr);
```
调用shmctl函数删除共享内存，或更复杂的操作

```
int shmctrl(int shmid, int cmd, struct shmid_sh *buf);
shmctrl(shmid, IPC_RMID, 0);
```



