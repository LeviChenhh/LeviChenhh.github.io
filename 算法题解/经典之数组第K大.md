# 经典之数组第K大

## 215. 数组中的第K个最大元素
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

>https://leetcode-cn.com/problems/kth-largest-element-in-an-array/

快速选择算法O(n)

```
class Solution {
public:
    int quickSelect(vector<int>& nums, int left, int right, int index)
    {
        int q = randomPartition(nums, left, right);
        if(q == index)
        {
            return nums[q];
        } else
        {
            return q < index ? quickSelect(nums, q + 1, right, index) : quickSelect(nums, left, q - 1, index);
        }
    }

    int randomPartition(vector<int>& nums, int left, int right)
    {
        int randP = rand() % (right - left + 1) + left;
        swap(nums[randP], nums[right]);
        return Partition(nums, left, right);
    }

    int Partition(vector<int>& nums, int left, int right)
    {
        int x = nums[right], i = left; // i 待放<=x的值，最后返回i-1
        for(int j = left; j <= right; j++)
        {
            if(nums[j] <= x)
            {
                swap(nums[i++], nums[j]);
            }
        }
        return i - 1;
    }
    int findKthLargest(vector<int>& nums, int k) {
        srand(time(0));
        return quickSelect(nums, 0, nums.size() - 1, nums.size() - k);
    }
};
```


堆排序（大根堆）
```
class Solution {
public:
    void maxHeapify(vector<int>& a, int i, int heapSize) {
        int l = i * 2 + 1, r = i * 2 + 2, largest = i;
        if (l < heapSize && a[l] > a[largest]) {
            largest = l;
        } 
        if (r < heapSize && a[r] > a[largest]) {
            largest = r;
        }
        if (largest != i) {
            swap(a[i], a[largest]);
            maxHeapify(a, largest, heapSize);
        }
    }

    void buildMaxHeap(vector<int>& a, int heapSize) {
        for (int i = heapSize / 2; i >= 0; --i) {
            maxHeapify(a, i, heapSize);
        } 
    }

    int findKthLargest(vector<int>& nums, int k) {
        int heapSize = nums.size();
        buildMaxHeap(nums, heapSize);
        for (int i = nums.size() - 1; i >= nums.size() - k + 1; --i) {
            swap(nums[0], nums[i]);
            --heapSize;
            maxHeapify(nums, 0, heapSize);
        }
        return nums[0];
    }
};
```