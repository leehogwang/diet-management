import 'package:flutter/material.dart';
import 'main_navigation_screen.dart';

class Yoon1Screen extends StatefulWidget {
  const Yoon1Screen({super.key});

  @override
  State<Yoon1Screen> createState() => _Yoon1ScreenState();
}

class _Yoon1ScreenState extends State<Yoon1Screen> {
  @override
  void initState() {
    super.initState();
    // 2초 후 자동으로 메인 화면으로 이동
    _navigateToMain();
  }

  void _navigateToMain() async {
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const MainNavigationScreen()),
      );
    }
  }

  void _navigateToMainNow() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const MainNavigationScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue.shade50,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // 아이콘
              Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: Colors.blue,
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.bar_chart,
                  size: 60,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 32),

              // 제목
              const Text(
                'Yoon-1 Features',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),

              // 설명
              const Text(
                '영양 정보 모니터링 시스템',
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.grey,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),

              // 기능 목록
              Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    children: [
                      _buildFeatureItem('칼로리 추적', Icons.local_fire_department),
                      const Divider(),
                      _buildFeatureItem('영양소 분석', Icons.analytics),
                      const Divider(),
                      _buildFeatureItem('데이터 시각화', Icons.show_chart),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 32),

              // 로딩 텍스트
              const Text(
                '잠시 후 메인 화면으로 이동합니다...',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),

              // 건너뛰기 버튼
              TextButton(
                onPressed: _navigateToMainNow,
                child: const Text(
                  '바로 가기',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.blue,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),

              // 진행 상황 표시
              const SizedBox(height: 16),
              const LinearProgressIndicator(
                backgroundColor: Colors.grey,
                valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureItem(String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        children: [
          Icon(
            icon,
            color: Colors.blue,
            size: 24,
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              title,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          const Icon(
            Icons.check_circle_outline,
            color: Colors.green,
            size: 20,
          ),
        ],
      ),
    );
  }
}