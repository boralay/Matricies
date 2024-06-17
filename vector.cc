#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "kernels.h"
#include <cassert>
#include <cmath>
using namespace std;
#define TXT(x)  #x << "=" << x
#define PRINT(x, s)  {cout << "variable " << #x << endl; printBytes(x, s);}


void printBytes(void* ptr, int m_size);

class AllocatorLeaking {
    public:
    // 8 byte aligned
    void* allocate(size_t s) {
        s = alignSize(s);
        assert(s <= pageSize);
        if (s >= spaceLeft) {
            ptr = reinterpret_cast<char*>(malloc(pageSize));
            std::cout << "allocated " << pageSize << " amount of memory at=" << (void*)ptr << std::endl;
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
    const size_t pageSize = 1024 * 1024 * 1024;
    size_t spaceLeft = 0;
    char* ptr = nullptr;   
    
    size_t alignSize(size_t size) {
        if(size & 0x7) return (size/8*8) + 8;
        else return size;
    }
};

static AllocatorLeaking singleton_allocator;
void* operator new (std::size_t count) {
    return singleton_allocator.allocate(count);
};
void* operator new[] (std::size_t count) {
    return singleton_allocator.allocate(count);
};
void operator delete (void* ptr) { 
    singleton_allocator.free(ptr);
};
void operator delete[] (void* ptr) { 
    singleton_allocator.free(ptr);
};




class Vector;
class AccessVector;
class ConstAccessVector {
    protected:
    size_t m_size = 0;
    double* ptr = nullptr;
    public:
    ConstAccessVector subvector(int begin_idx, int end_idx) const;
    int size() const {
        return m_size;
    }
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
    double operator[](int index) const {
        return this->at(index);
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
    friend AccessVector;
};

class AccessVector : public ConstAccessVector {
    public:
    using ConstAccessVector::subvector;
    AccessVector subvector(int begin_idx, int end_idx) {
        AccessVector temp;
        assert((end_idx - begin_idx) <= m_size);
        assert(begin_idx >= 0);
        assert(begin_idx <= m_size);
        assert(end_idx <= m_size);
        temp.ptr = this->ptr + begin_idx;
        temp.m_size = end_idx - begin_idx;
        return temp;
    }
    using ConstAccessVector :: at;
    double& at(int index) {
        assert(index < m_size);
        assert(index >= 0);
        return ptr[index];
    }
    double& operator[](int index) {
        return this->at(index);
    }
    AccessVector& operator+=(const ConstAccessVector& vec) {
        assert(vec.size() == m_size);
        for (int ii = 0; ii < m_size; ii++) {
            this->at(ii) += vec.at(ii);
        }
        return *this;
    }

    AccessVector& addOnGPU(const ConstAccessVector& vec) {
      void* this_on_device;
      void* other_on_device;
      assert(vec.size() == this->size());
      cudaMalloc(&this_on_device, sizeof(double)*this->size());
      cudaMalloc(&other_on_device, sizeof(double)*vec.size());
      cudaMemcpy(this_on_device, this->ptr, sizeof(double)*this->size(), cudaMemcpyHostToDevice);
      cudaMemcpy(other_on_device, vec.ptr, sizeof(double)*vec.size(), cudaMemcpyHostToDevice);

      // const int BLOCKSIZE = 256;
      //int numblocks = (this->size()+255)/256;
      //kernel_gpu_add_double_vectors<<<numblocks, BLOCKSIZE>>>(this->size(), (double*) this_on_device, (double*) other_on_device);

      kernel_gpu_add_double_vectors(this->size(), (double*) this_on_device, (double*) other_on_device);

      cudaMemcpy(this_on_device, this->ptr, sizeof(double)*this->size(), cudaMemcpyDeviceToHost);
      
      cudaFree(this_on_device);
      cudaFree(other_on_device);

     

      // allocate space on gpu using cuda_malloc
      // copy contents of both vectors onto gpu memory using cuda memcpy
      // execute a kernel on gpu to do the addition
      // bring back result to cpu using cuda memcpy
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
ConstAccessVector ConstAccessVector::subvector(int begin_idx, int end_idx) const {
    // code copy from AccessVector
    ConstAccessVector temp;
    assert((end_idx - begin_idx) <= m_size);
    assert(begin_idx >= 0);
    assert(begin_idx <= m_size);
    assert(end_idx <= m_size);
    temp.ptr = this->ptr + begin_idx;
    temp.m_size = end_idx - begin_idx;
    return temp;
}


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
    temp *= scalar;
    return temp;
}
Vector ConstAccessVector::operator-() const {
    Vector temp(*this);
    return temp * -1;
}

class Matrix : Vector {
    public:
    Matrix(int rows, int cols) {
        num_cols_ = cols;
        resize(rows, cols);
    }
    int num_cols() const {
        return num_cols_;
    }
    int num_rows() const {
        assert(num_cols() > 0);
        return Vector::size() / num_cols();
    }
    AccessVector operator[](int row) {
        return Vector::subvector(num_cols() * row, num_cols() * (row + 1));
    }
    ConstAccessVector operator[](int val) const;
    Vector operator*(ConstAccessVector& v) {
        assert(v.size() == num_cols());
        Vector v1(num_rows());
	for(int ii = 0; ii < num_rows(); ii++) {
            v1[ii] = (*this)[ii].dotProduct(v);
        }
        return v1;
    }
    void resize(int rows, int cols) {
        assert(rows > 0 && cols > 0);
        int product = rows*cols;
        Vector::resize(product);
        assert(product == Vector::size());
    }
    Matrix operator+(Matrix& m) {
        Matrix m1(*this);
        for (int ii = 0; ii < num_rows(); ii++) {
            m1[ii] = (*this)[ii] + m[ii];
        }
        return m1;
    }
    AccessVector getCol(int col) const {
        Vector v(num_rows());
        for (int ii = 0; ii < num_rows(); ii++) {
            v[ii] = (*this)[(ii * num_cols())][col];
        }
        return v;
    }
    Matrix operator*(Matrix& m) {
        assert(m.num_cols() == this->num_cols());
        assert(m.num_rows() == this->num_rows());
        Matrix m1(*this);
        for (int ii = 0; ii < num_rows(); ii++) {
            for (int jj = 0; jj < num_cols(); jj++) {
                m1[ii][jj] = (*this)[ii].dotProduct(m.getCol(jj));
            }
        } 
        return m1;
    }
    private: 
    int num_cols_ = 0;
};


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
void test_vector(Vector& v) {
    auto av7 = v.subvector(0, 3);
    Vector v8(av7);
    ConstAccessVector av9 = av7.subvector(2, 3);
    cout << TXT(v8) << endl;
    cout << TXT(av9) << endl;
    cout << TXT(av7) << endl;
    v.resize(20);
    v.at(0) = 5;
    cout << TXT(v) << endl;
    cout << TXT(av7) << endl;
    cout << TXT(v8) << endl;
    cout << TXT(av9) << endl;
    Vector v10(av7);
    v10.at(0) = 1.0;
    cout << TXT(v10) << endl;
    cout << TXT(av7) << endl;
}
void test_matrix() {
    Matrix M(2, 3);
    cout << M[0] << endl;
    M[0][0] = 2.1;
    M[1][1] = 3.1;
    M[1][2] = 4.1;
    auto v = M[1]; 
    auto v2 = M*v;
    cout << TXT(v) << endl;
    cout << TXT(v2) << endl;
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
    Vector v5 = v4 - v3;
    assert(v5 == v2);
    Vector v6 = v2 * 2;
    v2 += v2;
    assert(v6 == v2);
    cout << TXT(v2) << endl;
    cout << TXT(v3) << endl;
    cout << TXT(v4) << endl;
    cout << TXT(v5) << endl;
    cout << TXT(v6) << endl;
    test_vector(v2);
}
int main() {
    test_matrix();
}

void printBytes(void* ptr, int m_size) {
    for (int ii = 0; ii < m_size; ii++) {
        unsigned char* pointer = reinterpret_cast<unsigned char*>(ptr);
        int var = *(pointer + ii);
        cout << ii << ": " << var << endl;
    }
}


