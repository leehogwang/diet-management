import 'package:flutter/material.dart';
import 'yoon1_screen.dart';

class Patch1Screen extends StatefulWidget {
  const Patch1Screen({super.key});

  @override
  State<Patch1Screen> createState() => _Patch1ScreenState();
}

class _Patch1ScreenState extends State<Patch1Screen> {
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
        MaterialPageRoute(builder: (context) => const Yoon1Screen()),
      );
    }
  }

  void _navigateToMainNow() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const Yoon1Screen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.orange.shade50,
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
                  color: Colors.orange,
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.settings_suggest,
                  size: 60,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 32),

              // 제목
              const Text(
                'Patch-1 Features',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Colors.orange,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),

              // 설명
              const Text(
                '개인 설정 및 데이터 관리',
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
                      _buildFeatureItem('데이터 가져오기', Icons.download),
                      const Divider(),
                      _buildFeatureItem('설정 관리', Icons.settings),
                      const Divider(),
                      _buildFeatureItem('CSV 데이터 처리', Icons.table_chart),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 32),

              // 로딩 텍스트
              const Text(
                '잠시 후 Yoon-1 화면으로 이동합니다...',
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
                    color: Colors.orange,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),

              // 진행 상황 표시
              const SizedBox(height: 16),
              const LinearProgressIndicator(
                backgroundColor: Colors.grey,
                valueColor: AlwaysStoppedAnimation<Color>(Colors.orange),
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
            color: Colors.orange,
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