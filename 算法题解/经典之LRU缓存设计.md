# 经典之LRU缓存设计
## 146. LRU 缓存

请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：

LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。

函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

```
struct DListNode
{
    int key;
    int value;
    DListNode *prev;
    DListNode *next;
    DListNode(): key(0), value(0), prev(nullptr), next(nullptr){}
    DListNode(int _key, int _value): key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DListNode*> m_cacheMap;
    int m_iSize;
    int m_iCapacity;
    DListNode *head;
    DListNode *tail;
public:
    LRUCache(int capacity): m_iCapacity(capacity), m_iSize(0) {
        head = new DListNode();
        tail = new DListNode();
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        if(m_cacheMap.find(key) == m_cacheMap.end())
        {
            return -1;
        }

        DListNode *node = m_cacheMap[key];
        moveToHead(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if(m_cacheMap.find(key) == m_cacheMap.end())
        {
            DListNode *node = new DListNode(key, value);
            addToHead(node);
            m_cacheMap[key] = node;
            m_iSize++;
            if(m_iSize > m_iCapacity)
            {
                DListNode *removed = removetail();
                m_cacheMap.erase(removed->key);
                delete removed;
            }
        }
        else   // 找到更新加移动到头
        {
            DListNode * node = m_cacheMap[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void removeNode(DListNode *node)
    {
        node->next->prev = node->prev;
        node->prev->next = node->next;
    }

    void moveToHead(DListNode * node)
    {
        removeNode(node);
        addToHead(node);
    }

    void addToHead(DListNode * node)
    {
        DListNode * nexHead = head->next;
        head->next = node;
        node->next = nexHead;
        nexHead->prev = node;
        node->prev = head;
    }

    DListNode *removetail()
    {
        DListNode *preTail = tail->prev;
        removeNode(preTail);
        return preTail;
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```