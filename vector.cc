#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdlib.h>
#include "kernels.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
using namespace std;
#define EPSILON 1e-13
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

#ifdef LEAKY 
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
#endif


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

  AccessVector& computeOnGPU(const ConstAccessVector& vec, func_idpdp ff) {
      void* this_on_device;
      void* other_on_device;
      assert(vec.size() == this->size());
      cudaMalloc(&this_on_device, sizeof(double)*this->size());
      cudaMalloc(&other_on_device, sizeof(double)*vec.size());
      auto err1 = cudaMemcpy(this_on_device, this->ptr, sizeof(double)*this->size(), cudaMemcpyHostToDevice);
      if (err1 != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
      auto err2 = cudaMemcpy(other_on_device, vec.ptr, sizeof(double)*vec.size(), cudaMemcpyHostToDevice);
      if (err2 != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
      
      // const int BLOCKSIZE = 256;
      //int numblocks = (this->size()+255)/256;
      //kernel_gpu_add_double_vectors<<<numblocks, BLOCKSIZE>>>(this->size(), (double*) this_on_device, (double*) other_on_device);

      //   kernel_gpu_add_double_vectors(this->size(), (double*) this_on_device, (double*) other_on_device);
      (*ff)(this->size(), (double*) this_on_device, (double*) other_on_device);
      
      
      auto err = cudaMemcpy(this->ptr, this_on_device, sizeof(double)*this->size(), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
			    
      
      cudaFree(this_on_device);
      cudaFree(other_on_device);

      return *this;

      // allocate space on gpu using cuda_malloc
      // copy contents of both vectors onto gpu memory using cuda memcpy
      // execute a kernel on gpu to do the addition
      // bring back result to cpu using cuda memcpy
    } 
   AccessVector& addOnGPU(const ConstAccessVector& vec) {
     return computeOnGPU(vec, kernel_gpu_add_double_vectors);
   }
  double dotProductOnGPU(const ConstAccessVector& vec) const;
  
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
      // cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
    };
    // Vector(const Vector& other) {
    //    cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
    //    this->resize(other.size());
    // };
    Vector(const ConstAccessVector& other) {
      //  cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        construct(other);
    }
    Vector(const AccessVector& other) {
      //  cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        construct(other);
    }
    Vector(const Vector& other) {
      //   cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        construct(other);
    }
    explicit Vector(int sz) {
      //  cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        resize(sz);
    }
    ~Vector() {
      //        cout << "I am in " << __PRETTY_FUNCTION__ << TXT(this) << "\n";
        resize(0);
    }
    void construct(const ConstAccessVector& other) {
        this->resize(other.size());
        for (int ii = 0; ii < other.size(); ii++) {
            this->at(ii) = other.at(ii);
        }
    }
    void fillRandom() {
      for (int i = 0; i < this->size(); i++) {
	this->at(i) = drand48();
      }
    }
  void fillOnes() {
    for (int i = 0; i < this->size(); i++) {
	this->at(i) = 1;
      }
  }
  void fillConsecutive() {
    for (int i = 0; i < this->size(); i++) {
	this->at(i) = i;
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
double AccessVector::dotProductOnGPU(const ConstAccessVector& vec) const {
    Vector v1(*this);
    return v1.computeOnGPU(vec, kernel_gpu_dot_product_vectors)[0];
}

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

class Matrix : public Vector {
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
  ConstAccessVector operator[](int val) const {
    return const_cast<Matrix&>(*this)[val];
  }
  Vector operator*(ConstAccessVector& v) {
    assert(v.size() == num_cols());
    Vector v1(num_rows());
    for(int ii = 0; ii < num_rows(); ii++) {
      v1[ii] = (*this)[ii].dotProduct(v);
    }
    return v1;
  }
  friend ostream& operator<<(ostream& os, const Matrix& m) {
    os << "class Vector address: " << (&m) << " pointer: " << m.ptr << "\n";
    for (int ii = 0; ii < m.num_rows(); ii++) {
      for (int jj = 0; jj < m.num_cols(); jj++) {
	os << m[ii][jj] << " ";
      }
      os << "\n";
    }
    return os;
  }
  double& at(int row, int col) {
    return Vector::operator[](row * num_cols() + col);
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
  Matrix transpose() {
    Matrix m(this->num_cols(), this->num_rows());
    for (int row = 0; row < this->num_rows(); row++) {
      auto v = (*this)[row];
      for (int col = 0; col < v.size(); col++) {
	m.at(col, row) = v[col];
      }
    }
    return m;
  }
    AccessVector getCol(int col) const {
        Vector v(num_rows());
        for (int ii = 0; ii < num_rows(); ii++) {
            v[ii] = (*this)[ii][col];
        }
        return v;
    }
    Matrix operator*(Matrix& m) {
        assert(m.num_rows() == this->num_cols());
	Matrix m1(this->num_rows(), m.num_cols());
        for (int ii = 0; ii < num_rows(); ii++) {
            for (int jj = 0; jj < m.num_cols(); jj++) {
                m1[ii][jj] = (*this)[ii].dotProduct(m.getCol(jj));
            }
        } 
        return m1;
    }


    
    
  Matrix multiplyOnGPU(Matrix& other, bool RbyC) {
      void* transposed_matrix_on_device;
      void* other_on_device;
      void* result_on_device;
      void* this_on_device;
      void* transposed_other_on_device;
      assert(this->num_cols() == other.num_rows());
      auto transposed_matrix = this->transpose();
      auto transposed_other = other.transpose();
      cudaMalloc(&transposed_matrix_on_device, sizeof(double)*this->num_cols()*this->num_rows());
      cudaMalloc(&transposed_other_on_device, sizeof(double)*other.num_cols()*other.num_rows());
      cudaMalloc(&other_on_device, sizeof(double)*other.num_cols()*other.num_rows());
      cudaMalloc(&result_on_device, sizeof(double)*this->num_rows()*other.num_cols());
      cudaMalloc(&this_on_device, sizeof(double)*this->num_cols()*this->num_rows());
      auto err1 = cudaMemcpy(transposed_matrix_on_device, transposed_matrix.ptr, sizeof(double)*this->num_cols()*this->num_rows(), cudaMemcpyHostToDevice);
      if (err1 != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
      auto err2 = cudaMemcpy(other_on_device, other.ptr, sizeof(double)*other.num_cols()*other.num_rows(), cudaMemcpyHostToDevice);
      if (err2 != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
      auto err3 = cudaMemcpy(this_on_device, this->ptr, sizeof(double)*this->num_cols()*this->num_rows(), cudaMemcpyHostToDevice);
      if (err3 != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
      auto err4 = cudaMemset(result_on_device, 0, sizeof(double)*this->num_rows()*other.num_cols());
      if (err4 != cudaSuccess) {
	cout << "cudaMemset fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
      auto err6 = cudaMemcpy(transposed_other_on_device, transposed_other.ptr, sizeof(double)*other.num_cols()*other.num_rows(), cudaMemcpyHostToDevice);
      if (err6 != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" <<__FILE__ << ":" <<  __LINE__<< endl;
      }
      
      
      if (RbyC) {
	kernel_gpu_multiply_matrix_by_matrix_RbyC(this->num_rows(), other.num_rows(), other.num_cols(), (double*) this_on_device, (double*) transposed_other_on_device, (double*) result_on_device);
      } else {
	kernel_gpu_multiply_matrix_by_matrix_CbyR(this->num_rows(), other.num_rows(), other.num_cols(), (double*) transposed_matrix_on_device, (double*) other_on_device, (double*) result_on_device);
      }
      Matrix result(this->num_rows(), other.num_cols());
      auto err5 = cudaMemcpy(result.ptr, result_on_device, sizeof(double)*this->num_rows()*other.num_cols(), cudaMemcpyDeviceToHost);
      if (err5 != cudaSuccess) {
	cout << "cudaMemcpy fail at line:" << __FILE__ << ":" << __LINE__ << endl;
      }
      return result;
  }
    private: 
    int num_cols_ = 0;
};
bool isAboutSame(const double& double1, const double& double2) {
    if (double1 + double2 == 0) {
      return (double1 == 0);
    } 
    double comparison = fabs(double1 - double2) / fabs(double1 + double2); 
    if (comparison < EPSILON) {
      return true;
    } else {
      return false;
    }
}

bool isAboutSame(const ConstAccessVector& vec, const ConstAccessVector& vec2) {
  assert(vec.size() == vec2.size());
  for (int i = 0; i < vec.size(); i++) {
    if (!isAboutSame(vec[i], vec2[i])) {
      cout << setprecision(17) << TXT(i) << TXT(vec[i]) << " is not equal to " << TXT(vec2[i]) << endl;
      return false;
    }
  }
  return true;
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
   
    M[0][0] = 2.1;
    M[1][1] = 3.1;
    M[1][2] = 4.1;
    auto v = M[0];
    auto v2 = M[1];
    
    
    
    Matrix Mat(3, 5);
    Mat[0][0] = 5;
    Mat[1][1] = 3;
    Mat[1][2] = 4;
    auto result = M.multiplyOnGPU(Mat, 0);
    cout << M << endl;
    cout << Mat << endl;
    auto result_on_CPU = M*Mat;
    cout << result_on_CPU << endl;
    cout << result << endl;
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
void test3() {
  Vector v;
  v.resize(10);
  v.at(0) = 3;
  v.at(4) = 4;
  v.at(5) = 2;
  Vector v2;
  v2.resize(10);
  v2.at(0) = 5;
  v2.at(4) = 2;
  
  auto result = v.dotProductOnGPU(v2);
  cout << TXT(v) << endl;
  
  cout << TXT(v2) << endl;
  cout << result << endl;
}
void test5() {
  int K = 512;
  Matrix m(1, K);
  //m[0][0] = 1;
  //m[0][1] = 10;
  //m[0][2] = 100;
  //m[1][0] = 1000;
  //m[1][1] = 10000;
  //m[1][2] = 100000;
  //m.fillConsecutive();
  m.fillRandom();
  cout << TXT(m) << endl;
  Matrix m2(K, 1);
  //m2[0][0] = 2;
  //m2[0][1] = 3;
  m2.fillRandom();
  //  m2[1][0] = 4;
  //m2[1][1] = 5;
  //m2[2][0] = 6;
  //m2[2][1] = 7;
  cout << TXT(m2) << endl;
  auto res1 = m * m2;
  auto res2 = m.multiplyOnGPU(m2, 1);
  cout << TXT(res1) << endl;
  cout << TXT(res2) << endl;
  cout << TXT(isAboutSame(res1, res2)) << endl;
}

void test6() {
  int K = 500;
  Matrix m(100, K);
  m.fillRandom();
  Matrix m2(K, 103);
  m2.fillRandom();
  auto m3 = m * m2;
  auto m4 = m.multiplyOnGPU(m2, 1);
  auto m5 = m.multiplyOnGPU(m2, 0);
  cout << TXT(isAboutSame(m3, m4)) << endl;
  cout << TXT(isAboutSame(m3, m5)) << endl;
}
int main() {
   test6();
}


void printBytes(void* ptr, int m_size) {
    for (int ii = 0; ii < m_size; ii++) {
        unsigned char* pointer = reinterpret_cast<unsigned char*>(ptr);
        int var = *(pointer + ii);
        cout << ii << ": " << var << endl;
    }
}


