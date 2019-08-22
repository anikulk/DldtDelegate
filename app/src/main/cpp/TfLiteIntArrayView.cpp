//
// Created by anishak on 8/19/19.
//

class TfLiteIntArrayView {
 public:
  // Construct a view of a TfLiteIntArray*. Note, `int_array` should be non-null
  // and this view does not take ownership of it.
  explicit TfLiteIntArrayView(const TfLiteIntArray* int_array)
      : int_array_(int_array) {}

  typedef const int* const_iterator;
  const_iterator begin() const { return int_array_->data; }
  const_iterator end() const { return &int_array_->data[int_array_->size]; }

  TfLiteIntArrayView(const TfLiteIntArrayView&) = default;
  TfLiteIntArrayView& operator=(const TfLiteIntArrayView& rhs) = default;

 private:
  const TfLiteIntArray* int_array_;
};

