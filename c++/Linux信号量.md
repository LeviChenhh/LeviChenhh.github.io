# Linux信号量

信号量是一个整型变量，是一个非负数的计数器，给共享资源建立一个标志，表示该共享资源被占用情况。
可以对其执行 P 和 V 操作。
P申请资源：如果信号量大于零，就对其进行减 1 操作；如果信号量等于 0，进程进入 waiting 状态，等待信号量大于零。
V释放资源：对信号量执行加 1 操作，并唤醒正在 waiting 的进程
二值信号量只能取 0 或者 1，是信号量的一种特殊形式，表示资源只有可用和不可用两种状态。与互斥量类似，、可以理解成加锁解锁操作，0 表示已经加锁，1 表示解锁。

信号量的操作
```
int semget(key_t key, int nsems, int semflg);    //用于获取或者创建信号量

//信号量的初始化不能直接用semget(key, 1, 0666|IPC_CREAT)，因为信号量创建后，初始值是0
//信号量的初始分三个步骤
//1)获取信号量，如果成功，函数返回
//2）如果失败，则创建信号量
//3）设置信号量的初始值。

//获取信号量
key_t key = 5005;
if((semid = semget(key, 1, 0666)) == -1)
{
    //如果信号量不存在，创建它
    if(errno == 2)
    {
        //用IPC_EXCL标志确保只有一个进程创建并初始化信号量，其他进程只能获取。
        if((semid = semget(key, 1, 0666|IPC_CREAT|IPC_EXCL)) == -1)
        {
            if(errno != EEXIST)
            {
                perror("int 1 semget()"); return false
            }
            if((semid = semget(key, 1, 0666)) == -1)
            {
                perror("init 2 semget()"); return false;
            }
            return true;
        }
        
        //信号量创建成功后，还需要把它初始化成value
        union semun sem_union;
        sem_union.val = value;    //设置信号量的初始值
        if(semctl(semid, 0, SETVAL, sem_union) < 0)
        {
            peeor("init semctrl()");
            return false;
        }
    }
    else
    {
        perror("init 3 semget()");
        return false;
    }
}


semop(m_semid,&sem_b,1)
//如果把sem_glg设置为SEM_UNDO，操作系统会跟踪进程对信号量的修改情况
//在全部的进程(正常或异常)终止后，操作系统将会把信号量恢复为初始值（就像撤销了全部进程对信号的操作）。
//如果信号量用于表示可用资源的数量，设置为SEM_UNDO更适合。
//如果信号量用于生产消费者模型，设置为0更合适。
```

