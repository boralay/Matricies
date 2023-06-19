#include <iostream>
#include <cmath>
using namespace std;
#define TXT(x)  #x << "=" << x
#define PRINT(x, s)  {cout << "variable " << #x << endl; printBytes(x, s);}

void printBytes(void* ptr, int m_size);

// todo: read about debugger and makefile capailities of tool.
// virtual function and runtime polymorphism

class AllocatorLeaking {
    public:
    // 8 byte aligned
    const size_t pageSize = 1024 * 1024 * 1024;
    void* allocate(size_t s) {
        s = alignSize(s);
        assert(s <= pageSize);
        if (s >= spaceLeft) {
            ptr = reinterpret_cast<char*>(malloc(pageSize));
            spaceLeft = pageSize;
        }
        char* temp = ptr;
        ptr += s;
        spaceLeft -= s;
        return temp;
    }
    void free(void* ptr){ 

    }
    private:
    size_t spaceLeft = 0;
    char* ptr = nullptr;
    size_t alignSize(size_t size) {
        int num = size >> 3 << 3;
        // return num % 8 ? num + 8 : num;
        assert(num >= size);
        assert(num % 8 == 0);
        assert(num < size + 8);
        return num & 0x7 ? num + 8 : num;
    }
};

// static AllocatorLeaking singleton_allocator;
// void* operator new (std::size_t count) {
//     return singleton_allocator.allocate(count);
// };
// void* operator new[] (std::size_t count) {
//     return singleton_allocator.allocate(count);
// };
// void operator delete (void* ptr) { 
//     singleton_allocator.free(ptr);
// };
// void operator delete[] (void* ptr) { 
//     singleton_allocator.free(ptr);
// };
class Vector;
class AccessVector;
class ConstAccessVector {
    protected:
    size_t m_size = 0;
    double* ptr = nullptr;
    public:
    int size() const {
        return m_size;
    };
    friend ostream& operator<<(ostream& os, const ConstAccessVector& ob) {
        os << "class Vector address: " << (&ob) << ". size: " << ob.m_size << ". pointer: " << ob.ptr << "\n";
        for (int ii = 0; ii < ob.m_size; ii++) {
            os << "Element " << ii << ": " << ob.ptr[ii] << "\n";
        }
        return os;
    }
    double at(int index) const {
        assert(index < m_size);
        assert(index >= 0);
        return ptr[index];
    }
    double magnitude() const {
        return sqrt(this->dotProduct(*this));
    };
    double dotProduct(const ConstAccessVector& vec2) const {
        double sum = 0;
        for (int ii = 0; ii < m_size; ii++) {
            sum += (this->at(ii) * vec2.at(ii));
        }
        return sum;
    };
    bool operator==(const ConstAccessVector& other) const {
        if (other.size() != this->size()) {
            return false;
        }
        for (int ii = 0; ii < other.size(); ii++) {
            if (other.at(ii) != this->at(ii)) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const ConstAccessVector& other) const {
        return !(*this == other);
    }
    Vector operator-() const;
    Vector operator+(const ConstAccessVector& vec) const;
    Vector operator-(const ConstAccessVector& vec) const;
    Vector operator*(double scalar) const;
};

class AccessVector : public ConstAccessVector {
    public:
    using ConstAccessVector :: at;
    double& at(int index) {
        assert(index < m_size);
        assert(index >= 0);
        return ptr[index];
    }
    AccessVector& operator+=(const ConstAccessVector& vec) {
        assert(vec.size() == m_size);
        for (int ii = 0; ii < m_size; ii++) {
            this->at(ii) += vec.at(ii);
        }
        return *this;
    }
    AccessVector& operator-=(const ConstAccessVector& vec) {
        assert(vec.size() == m_size);
        for (int ii = 0; ii < m_size; ii++) {
            this->at(ii) -= vec.at(ii);
        }
        return *this;
    }
    AccessVector& operator*=(double scalar) {
        for (int ii = 0; ii < m_size; ii++) {
            this->at(ii) *= scalar;
        }
        return *this;
    }
};

class Vector : public AccessVector {
    public:
    void pushVal(int index, double val) {
        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        if (m_size <= index) {
            resize(index + 1);
        }
        at(index) = val;
    }
    size_t alignment_shift() { return 3; } // config
    size_t alignment() { return 1 << alignment_shift(); }
    size_t upround_to_alignment(size_t sz) {  
        if ((sz & (alignment() - 1)) == 0) { 
            return sz;
        } else {
            return sz / alignment() * alignment() + alignment();
        }
    }
    void resize(int sz) {
        assert(sz >= 0);
        if (sz == m_size) {
            return;
        }
        assert(memory_size >= m_size);
        if (memory_size >= sz) {
            m_size = sz;
            return;
        }
        assert(memory_size < sz);
        allocate(upround_to_alignment(sz));
        m_size = sz;
        assert(m_size <= memory_size);
    }
    Vector() {
        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
    };
    // Vector(const Vector& other) {
    //    cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
    //    this->resize(other.size());
    // };
    Vector(const ConstAccessVector& other) {
        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        construct(other);
    }
    Vector(const AccessVector& other) {
        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        construct(other);
    }
    Vector(const Vector& other) {
        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        construct(other);
    }
    explicit Vector(int sz) {
        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        resize(sz);
    }
    ~Vector() {
        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        resize(0);
    }
    void construct(const ConstAccessVector& other) {
        this->resize(other.size());
        for (int ii = 0; ii < other.size(); ii++) {
            this->at(ii) = other.at(ii);
        }
    }
    private:
    size_t memory_size = 0;
    void allocate(int msz) {
        assert(msz >= 0);
        assert(msz > memory_size);
        double* temp = new double[msz];
        for (int ii = 0; ii < memory_size; ii++) {
            temp[ii] = ptr[ii];
        }
        delete[] ptr;
        ptr = temp;
        for (int ii = memory_size; ii < msz; ii++) {
            ptr[ii] = 0;
        }
        memory_size = msz;
    }
    // int mem2 = 0;
};
Vector ConstAccessVector::operator+(const ConstAccessVector& vec) const {
    Vector temp(*this);
    temp += vec;
    cout << "Temp: " << temp << endl;
    return temp;
}
Vector ConstAccessVector::operator-(const ConstAccessVector& vec) const {
    Vector temp(*this);
    return temp -= vec;
}
Vector ConstAccessVector::operator*(double scalar) const {
    Vector temp(*this);
    return temp *= scalar;
}
Vector ConstAccessVector::operator-() const {
    Vector temp(*this);
    return temp * -1;
}


void test1() {
    Vector obj;
    cout << obj << endl;
    Vector* pobj3 = new Vector;
    cout << *pobj3 << endl;
    obj.pushVal(0, 3.4);
    obj.pushVal(2, 4.3);
    cout << obj << endl;
    Vector obj2(4);
    obj2.at(3) = 3.1;
    obj2.pushVal(0, 3);
    obj2.resize(4);
    obj.resize(4);
    cout << TXT(obj) << endl;
    cout << TXT(obj2) << endl;
    Vector result = obj + obj2;
    cout << TXT(result) << endl;
}
void test2() {
    Vector v;
    v.resize(3);
    v.resize(2); 
    v.at(1) = 3;
    v.at(0) = 4;
    v.resize(5);
    v.resize(9);
    Vector v2(v);
    assert(v2 == v);
    cout << TXT(v2) << endl;
    Vector v3 = v2 * 3;
    assert(v3 != v2);
    double mag = v2.magnitude();
    assert(mag == 5);
    double dot = v2.dotProduct(v3);
    assert(dot == 75);
    Vector v4 = v2 + v3;
    assert(v4.magnitude() != 5 && v4.dotProduct(v3) != 75);

    
    cout << v3 << endl;
}
int main() {
    test2();
}

void printBytes(void* ptr, int m_size) {
    for (int ii = 0; ii < m_size; ii++) {
        unsigned char* pointer = reinterpret_cast<unsigned char*>(ptr);
        int var = *(pointer + ii);
        cout << ii << ": " << var << endl;
    }
}
// DONE : 

// todo: write class named AllocatorLeaking
// allocate big chunk of memory using malloc when new is called return from this chunk and if delete is called do nothing. if chunk finishes then allocate new chunk
// have static singleton variable of type AllocatorLeaking called singleton_allocator_leaking
// todo 2: define operator new and operator delete using singleton_allocator_leaking
// todo 3: memm_size member do not free already allocated memory when shrinking, only increase always to the power of 2. have funciton called 
// allocate which does what resize is doing
// todo 4: shrink to minimal possible memm_size at that time