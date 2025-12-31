import 'dart:io';
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../theme/app_colors.dart';

class NutriFoodScannerScreen extends StatefulWidget {
  final String palette;
  final bool isDarkMode;

  const NutriFoodScannerScreen({
    super.key,
    this.palette = 'sage',
    this.isDarkMode = false,
  });

  @override
  State<NutriFoodScannerScreen> createState() => _NutriFoodScannerScreenState();
}

class _NutriFoodScannerScreenState extends State<NutriFoodScannerScreen> {
  File? _selectedImage;
  bool _isAnalyzing = false;
  List<FoodResult>? _analysisResult;

  // Mock Food Database
  final List<FoodResult> _mockFoodDB = [
    FoodResult(
      name: '된장찌개',
      calories: 180,
      carbs: 15,
      protein: 12,
      fat: 8,
      sugar: 2,
      sodium: 800,
      fiber: 4,
    ),
    FoodResult(
      name: '잡곡밥',
      calories: 300,
      carbs: 65,
      protein: 6,
      fat: 1,
      sugar: 0,
      sodium: 5,
      fiber: 6,
    ),
    FoodResult(
      name: '제육볶음',
      calories: 450,
      carbs: 20,
      protein: 35,
      fat: 25,
      sugar: 8,
      sodium: 900,
      fiber: 2,
    ),
    FoodResult(
      name: '아보카도 샐러드',
      calories: 280,
      carbs: 12,
      protein: 8,
      fat: 22,
      sugar: 3,
      sodium: 100,
      fiber: 7,
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final theme = AppTheme.getTheme(widget.palette, widget.isDarkMode);
    final primaryColor = theme.colorScheme.primary;
    final cardColor = theme.colorScheme.surface;
    final textColor = theme.colorScheme.onSurface;

    return Theme(
      data: theme,
      child: Scaffold(
        backgroundColor: _selectedImage == null
            ? const Color(0xFF2C3E30)
            : theme.scaffoldBackgroundColor,
        body: SafeArea(
          child: Column(
            children: [
              if (_selectedImage == null)
                _buildCameraInterface(primaryColor)
              else if (_isAnalyzing)
                _buildAnalyzingInterface(primaryColor)
              else
                _buildResultInterface(theme, cardColor, textColor),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCameraInterface(Color primaryColor) {
    return Expanded(
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                IconButton(
                  icon: const Icon(Icons.close, color: Color(0xFFD4DCD6)),
                  onPressed: () => Navigator.pop(context),
                ),
                const Text(
                  'AI 스캐너',
                  style: TextStyle(
                    color: Color(0xFFD4DCD6),
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    letterSpacing: 1.5,
                  ),
                ),
                const SizedBox(width: 40), // Balance the close button
              ],
            ),
          ),

          // Camera frame
          Expanded(
            child: Center(
              child: Container(
                width: 288,
                height: 288,
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white.withOpacity(0.2),
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(40),
                ),
                child: Stack(
                  children: [
                    Positioned(
                      top: 16,
                      left: 16,
                      child: Container(
                        width: 16,
                        height: 16,
                        decoration: BoxDecoration(
                          border: Border(
                            top: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                            left: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      top: 16,
                      right: 16,
                      child: Container(
                        width: 16,
                        height: 16,
                        decoration: BoxDecoration(
                          border: Border(
                            top: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                            right: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      bottom: 16,
                      left: 16,
                      child: Container(
                        width: 16,
                        height: 16,
                        decoration: BoxDecoration(
                          border: Border(
                            bottom: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                            left: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      bottom: 16,
                      right: 16,
                      child: Container(
                        width: 16,
                        height: 16,
                        decoration: BoxDecoration(
                          border: Border(
                            bottom: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                            right: BorderSide(
                              color: const Color(0xFFD4DCD6),
                              width: 2,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Instruction text
          const Text(
            '음식을 프레임 안에 맞춰주세요',
            style: TextStyle(
              color: Color(0xFFD4DCD6),
              fontSize: 14,
              fontWeight: FontWeight.w300,
              letterSpacing: 0.5,
            ),
          ),

          const SizedBox(height: 32),

          // Capture button (simulated with image picker)
          SizedBox(
            height: 128,
            child: Center(
              child: GestureDetector(
                onTap: _pickImage,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: Colors.white.withOpacity(0.3),
                      width: 2,
                    ),
                  ),
                  child: Center(
                    child: Container(
                      width: 64,
                      height: 64,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.white.withOpacity(0.3),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAnalyzingInterface(Color primaryColor) {
    return Expanded(
      child: Stack(
        children: [
          // Background image with overlay
          if (_selectedImage != null)
            Positioned.fill(
              child: ColorFiltered(
                colorFilter: ColorFilter.mode(
                  Colors.black.withOpacity(0.6),
                  BlendMode.srcOver,
                ),
                child: Image.file(
                  _selectedImage!,
                  fit: BoxFit.cover,
                ),
              ),
            ),

          // Analyzing UI
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 64,
                  height: 64,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                    backgroundColor: Colors.white.withOpacity(0.2),
                  ),
                ),
                const SizedBox(height: 32),
                const Text(
                  '분석 중...',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w500,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  '재료를 식별하고 있습니다',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w300,
                    color: Color(0xFFD4DCD6),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultInterface(ThemeData theme, Color cardColor, Color textColor) {
    final primaryColor = theme.colorScheme.primary;

    return Expanded(
      child: Column(
        children: [
          // Image preview section
          SizedBox(
            height: MediaQuery.of(context).size.height * 0.4,
            child: Stack(
              children: [
                Positioned.fill(
                  child: _selectedImage != null
                      ? Image.file(
                          _selectedImage!,
                          fit: BoxFit.cover,
                        )
                      : Container(color: const Color(0xFF2C3E30)),
                ),
                Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        const Color(0xFF2C3E30).withOpacity(0),
                        const Color(0xFF2C3E30).withOpacity(0.8),
                      ],
                    ),
                  ),
                ),
                Positioned(
                  top: 24,
                  left: 24,
                  child: GestureDetector(
                    onTap: _resetScanner,
                    child: Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.2),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.arrow_back,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
                Positioned(
                  bottom: 32,
                  left: 32,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(
                            Icons.check_circle,
                            color: Color(0xFFD4DCD6),
                            size: 14,
                          ),
                          const SizedBox(width: 8),
                          const Text(
                            '분석 완료',
                            style: TextStyle(
                              color: Color(0xFFD4DCD6),
                              fontSize: 12,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1.5,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        '건강한 음식 2개\n발견됨',
                        style: TextStyle(
                          fontSize: 32,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          height: 1.2,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          // Results sheet
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: theme.scaffoldBackgroundColor,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(40),
                  topRight: Radius.circular(40),
                ),
              ),
              child: Column(
                children: [
                  // Handle
                  Container(
                    margin: const EdgeInsets.only(top: 8),
                    width: 48,
                    height: 4,
                    decoration: BoxDecoration(
                      color: const Color(0xFFD4DCD6),
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),

                  // Results list
                  Expanded(
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.all(32),
                      child: Column(
                        children: [
                          // Food items
                          ...(_analysisResult?.map((food) => _buildFoodCard(
                                food,
                                cardColor,
                                textColor,
                                primaryColor,
                              )) ?? []),

                          const SizedBox(height: 24),

                          // Add button
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton(
                              onPressed: () {
                                // Add to diet
                                Navigator.pop(context);
                              },
                              style: ElevatedButton.styleFrom(
                                backgroundColor: primaryColor,
                                foregroundColor: Colors.white,
                                padding: const EdgeInsets.symmetric(vertical: 20),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(24),
                                ),
                                elevation: 8,
                                shadowColor: Colors.black.withOpacity(0.2),
                              ),
                              child: const Text(
                                '식단에 추가하기',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFoodCard(FoodResult food, Color cardColor, Color textColor,
      Color primaryColor) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          // Icon
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: primaryColor,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Center(
              child: Text(
                food.name[0],
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
          ),

          const SizedBox(width: 20),

          // Info
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  food.name,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: textColor,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${food.calories} kcal • 탄수 ${food.carbs}g',
                  style: TextStyle(
                    fontSize: 12,
                    color: textColor.withOpacity(0.6),
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),

          // Add button
          GestureDetector(
            onTap: () {},
            child: Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                border: Border.all(
                  color: textColor.withOpacity(0.2),
                ),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.add,
                size: 16,
                color: Colors.grey,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _pickImage() async {
    // TODO: Implement actual image picker
    // For now, just simulate picking an image and analyzing

    setState(() {
      // Simulate image selection
      _isAnalyzing = true;
    });

    // Simulate analysis delay
    await Future.delayed(const Duration(seconds: 2));

    setState(() {
      _isAnalyzing = false;
      _analysisResult = [_mockFoodDB[3], _mockFoodDB[1]];
    });
  }

  void _resetScanner() {
    setState(() {
      _selectedImage = null;
      _analysisResult = null;
      _isAnalyzing = false;
    });
  }
}

class FoodResult {
  final String name;
  final int calories;
  final int carbs;
  final int protein;
  final int fat;
  final int sugar;
  final int sodium;
  final int fiber;

  FoodResult({
    required this.name,
    required this.calories,
    required this.carbs,
    required this.protein,
    required this.fat,
    required this.sugar,
    required this.sodium,
    required this.fiber,
  });
}
