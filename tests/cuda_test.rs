
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_transfer() {
    let mut cuda_storage = CudaStorage::<f32>::new(4, 4).unwrap();
    let host_input = vec![1.0f32; 16];
    cuda_storage.copy_from_host(&host_input).unwrap();
    
    let mut host_output = vec![0.0f32; 16];
    cuda_storage.copy_to_host(&mut host_output).unwrap();
    
    for i in 0..16 {
        assert_eq!(host_input[i], host_output[i]);
    }
}
