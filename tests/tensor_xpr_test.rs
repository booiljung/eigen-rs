use eigen_rs::core::tensor::Tensor;

#[test]
fn test_tensor_lazy_add() {
    let mut a = Tensor::<f64, 2>::new([2, 2]).unwrap();
    let mut b = Tensor::<f64, 2>::new([2, 2]).unwrap();

    *a.get_mut([0, 0]).unwrap() = 1.0;
    *a.get_mut([1, 1]).unwrap() = 2.0;
    *b.get_mut([0, 0]).unwrap() = 10.0;
    *b.get_mut([1, 1]).unwrap() = 20.0;

    let mut c = Tensor::<f64, 2>::new([2, 2]).unwrap();
    // Lazy addition
    let xpr = &a + &b;
    c.assign(&xpr).unwrap();

    assert_eq!(*c.get([0, 0]).unwrap(), 11.0);
    assert_eq!(*c.get([1, 1]).unwrap(), 22.0);
}

#[test]
fn test_tensor_chained_xpr() {
    let mut a = Tensor::<f64, 2>::new([2, 2]).unwrap();
    let mut b = Tensor::<f64, 2>::new([2, 2]).unwrap();

    *a.get_mut([0, 0]).unwrap() = 1.0;
    *b.get_mut([0, 0]).unwrap() = 10.0;

    let mut c = Tensor::<f64, 2>::new([2, 2]).unwrap();
    // Lazy (A + B) + A
    let xpr = (&a + &b) + &a;
    c.assign(&xpr).unwrap();

    assert_eq!(*c.get([0, 0]).unwrap(), 12.0);
}

#[test]
fn test_tensor_scalar_mul() {
    let mut a = Tensor::<f64, 2>::new([2, 2]).unwrap();
    *a.get_mut([0, 0]).unwrap() = 5.0;

    let mut b = Tensor::<f64, 2>::new([2, 2]).unwrap();
    // Lazy A * 2.0
    let xpr = &a * 2.0;
    b.assign(&xpr).unwrap();

    assert_eq!(*b.get([0, 0]).unwrap(), 10.0);
}

#[test]
fn test_tensor_complex_chain() {
    let mut a = Tensor::<f64, 2>::new([2, 2]).unwrap();
    let mut b = Tensor::<f64, 2>::new([2, 2]).unwrap();
    *a.get_mut([0, 0]).unwrap() = 1.0;
    *b.get_mut([0, 0]).unwrap() = 10.0;

    let mut c = Tensor::<f64, 2>::new([2, 2]).unwrap();
    // Lazy (A + B) * 3.0
    let xpr = (&a + &b) * 3.0;
    c.assign(&xpr).unwrap();

    assert_eq!(*c.get([0, 0]).unwrap(), 33.0);
}

#[test]
fn test_tensor_broadcasting_add() {
    let mut a = Tensor::<f64, 2>::new([2, 2]).unwrap();
    *a.get_mut([0, 0]).unwrap() = 1.0;

    let mut b = Tensor::<f64, 2>::new([2, 2]).unwrap();
    // Lazy A + 10.0
    let xpr = &a + 10.0;
    b.assign(&xpr).unwrap();

    assert_eq!(*b.get([0, 0]).unwrap(), 11.0);
    assert_eq!(*b.get([0, 1]).unwrap(), 10.0);
}

#[test]
fn test_tensor_broadcasting_sub() {
    let mut a = Tensor::<f64, 2>::new([2, 2]).unwrap();
    *a.get_mut([0, 0]).unwrap() = 5.0;

    let mut b = Tensor::<f64, 2>::new([2, 2]).unwrap();
    // Lazy A - 2.0
    let xpr = &a - 2.0;
    b.assign(&xpr).unwrap();

    assert_eq!(*b.get([0, 0]).unwrap(), 3.0);
    assert_eq!(*b.get([0, 1]).unwrap(), -2.0);
}
