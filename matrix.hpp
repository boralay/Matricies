namespace BMP {

  class Vector;
  class AccessVector;
  class ConstAccessVector;

  template<class T>
  class ConstAccessTVector {
    T* begin;
    T* end;

  public:
    typedef int Index;
    inline void init(const Vector&);  
    ConstAccessVector(const Vector&);
    ConstAccessVector(const ConstAccessVector&)=default;  
    void resize(Index, Index);
    ConstAccessVector(const AccessVector&);  
    ConstAccessVector() { begin=end=pointer=nullptr; }

    const T& operator * () const; //with asserts on range
    const T& operator [] (Index) const; //with asserts on range

    //use code written for AccessTVector for the functions below
    TVector operator + (const ConstAccessTVector&);
    TVector operator - (const ConstAccessTVector&);
    TVector operator * (const T&);
    T dot_product (const ConstAccessTVector&);

    T sum();
    T magnitude(); //norm assuming sqrt is defined on T
   
    friend ostream& (ostream&, const ConstAccessVector&);
  };



  template<class T>
  class AccessTVector : ConstAccessTVector {
  public:
    inline void init(TVector&);  
    AccessTVector(TVector&);  
    AccessTVector(const ConstAccessTVector&)=delete;  
    AccessTVector(const AccessTVector&) =default;  
    AccessTVector()=default;

    T& operator * () const; //with asserts on range
    T& operator [] (int) const; //with asserts on range


    AccessTVector& operator += (const ConstAccessTVector&);
    AccessTVector& operator -= (const ConstAccessTVector&);
    AccessTVector& operator *= (const T&);
    T dot_product (const ConstAccessTVector&);
   
    friend ostream& (ostream&, const AccessVector&);
  };


  template<class T>
  class TVector : private AccessTVector {
  public:
    inline void init(TVector&);  
    TVector(TVector&);
    TVector(const ConstAccessTVector&);  
    TVector(const AccessTVector&);  
    TVector(int size) {
      allocate(size);
    } //allocate and initialize to T() calling allocated which calls new[]
    TVector() { begin=end=pointer=nullptr; }

    TVector& operator= (const ConstAccessTVector&);
    TVector& operator= (const AccessTVector&);
    TVector& operator= (const TVector&);
   
    //ConstAccessVector(const ConstAccessTVector&)=delete;  
   
    T& operator [] (int);
    const T& operator [] (int) const;
    AccessTVector operator [] (int, int); //range
    ConstAccessTVector operator [] (int, int) const; //range

   
    void resize();
    ~TVector(); //free the memory
  protected:
    void allocate(Index sz) {
      begin = new[sz] T;
      end = begin + sz;
      pointer = nullptr;
    }

  };


  template<T>
  class ConstAccessTMatrix {
    Index num_cols;
    //similar to vector but with 2 index


    //and more:
    TVector operator * (const ConstAccessTVector&);
    TVector operator * (const ConstAccessTMatrix&);
    TMatrix<T> transpose();
    Tmatrix operator ^ (int); //implement -1, and other integers
    operator+();
  };

  template<T>
  class AccessTMatrix : ConstAccessTMatrix {
    //similar to vector but with 2 index
  };


  template<T>
  class TMatrix : private TVector<T>{
    Index num_rows=0;
    Index num_cols=0;
  public:
    explicit Tmatrix(const AccessTVector&);
    explicit TMatrix(const ConstAccessTVector&);
    explicit TMatrix(TVector&);  
    TMatrix(const TMatrix&);  
    TVector(int size); //allocate and initialize to T() calling allocated which calls new[]
    TVector() { begin=end=pointer=nullptr; }


    explicit operator = (const AccessTVector&);
    explicit operator = (const ConstAccessTVector&);
    explicit operator = (TVector&);

    void resize(Index, Index);  // using TVector::resize

    AccessTVector operator[] (Index); //returs a row
    AccessTVector row (Index); //returs a row
    AccessTVector column (Index); //returs a row
    T& operator[] (Index,Index);
    const T& operator[] (Index,Index) const;

   
    AccessTMatrix sub_matrix(Index r1,Index c1,Index r2, Index c2);
    ConstAccessTMatrix sub_matrix(Index r1,Index c1,Index r2, Index c2) const;

    friend ConstAccessTMatrix<T>;
    friend AccessTMatrix<T>;
  };
};

