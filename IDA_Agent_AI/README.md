# online_imitation_ai
강화학습을 활용한 자가학습 AI 모델

# 의존성 설치파일 [requirements.txt]
pip install -r requirements.txt

# 실행시 단계
conda init           # 활성화 설정 추가
conda init bash      # bash용 설정

conda create -n mouse_ai python=3.10       # 환경 생성 (1번만 하면 됨)

conda activate mouse_ai                    # 환경 활성화
pip install -r requirements.txt            # 라이브러리 설치
python demo_recorder.py --record           # 실행

# 깃허브 저장하는 명령어
git add .
git commit -m "모든 변경사항 커밋"
git push