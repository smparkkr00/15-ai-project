import torch

def check_pytorch():
    print("=" * 50)
    print("[1] PyTorch 기본 정보 확인")
    print(f"    - PyTorch 버전: {torch.__version__}")
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"    - CUDA 이용 가능 여부: {cuda_available}")
    
    if cuda_available:
        print(f"    - 내장 CUDA 버전: {torch.version.cuda}")
        print(f"    - 연결된 GPU 장치 수: {torch.cuda.device_count()}")
        print(f"    - 현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
        
        print("\n[2] GPU 텐서 연산 테스트")
        try:
            # 텐서를 GPU(cuda) 메모리에 생성
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            y = torch.tensor([4.0, 5.0, 6.0], device="cuda")
            
            # GPU 내부에서 더하기 연산 수행
            z = x + y
            
            print(f"    - 입력 x (GPU): {x}")
            print(f"    - 입력 y (GPU): {y}")
            print(f"    - 결과 z (GPU): {z}")
            print("\n🎉 축하합니다! GPU 가속 연산이 정상적으로 작동합니다.")
            
        except Exception as e:
            print(f"\n❌ GPU 연산 중 에러가 발생했습니다: {e}")
            
    else:
        print("\n[2] CPU 텐서 연산 테스트 (GPU 미연결)")
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"    - 결과 z (CPU): {z}")
        print("\n💡 PyTorch는 정상 작동하나, CPU 모드로 구동 중입니다.")
        print("   GPU를 사용하려면 NVIDIA 드라이버 상태나 PyTorch 설치 명령어를 확인해 주세요.")
        
    print("=" * 50)

if __name__ == "__main__":
    check_pytorch()