#[cfg(feature = "suitesparse")]
use eigen_rs::core::sparse::bridges::suitesparse::CholmodLLT;
#[cfg(feature = "mkl")]
use eigen_rs::core::sparse::bridges::mkl::MklPardiso;

#[test]
fn test_cholmod_bridge_init() {
    #[cfg(feature = "suitesparse")]
    {
        let _solver = CholmodLLT::<f64>::new();
    }
}

#[test]
fn test_mkl_bridge_init() {
    #[cfg(feature = "mkl")]
    {
        let _solver = MklPardiso::<f64>::new(true, true);
    }
}
