import bisect

class KNN_regressor:
    def __init__(self, k, x_train, y_train):
        assert(k < len(x_train))
        self.k = k
        self.data = list(zip(x_train, y_train))
        self.data.sort()
       
    
    def regress(self, x):
        """
            Chooses the nearest K elements by fisrt finding the locatino for
            x and then going left and right in order
            
            Since the data is sorted, we can use binary search which will take
            O(log(N)), and then we will have to go through the k elements which
            will take O(k) time. So the total time will be O(log(N) + k)
        """
        
        left = bisect.bisect_right(self.data, (x, 0))
        right = left + 1
        
        if left >= len(self.data):
            left = len(self.data) - 1
            
        sum_ = 0
        added = 0
        while (added < self.k):
            
            if left < 0:
                sum_ += self.data[right][1]
                right += 1  
            elif right >=  len(self.data):
                sum_ += self.data[left][1]
                left -= 1
            else:
            
                l_x, l_y = self.data[left]
                r_x, r_y = self.data[right]
                if abs(l_x - x) <= abs(r_x - x):
                    sum_ += l_y
                    left -= 1
                else:
                    sum_ += r_y
                    right += 1
                    
            added += 1
      
        return sum_ / added