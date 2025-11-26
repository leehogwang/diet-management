import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/food_database_service.dart';
import '../models/meal_nutrition.dart';
import '../database/database_helper.dart';

class FoodCameraScreen extends StatefulWidget {
  const FoodCameraScreen({super.key});

  @override
  State<FoodCameraScreen> createState() => _FoodCameraScreenState();
}

class _FoodCameraScreenState extends State<FoodCameraScreen> {
  final ImagePicker _picker = ImagePicker();
  final FoodDatabaseService _foodDb = FoodDatabaseService();

  XFile? _imageFile;
  Uint8List? _imageBytes;
  bool _isProcessing = false;
  String? _recognizedFood;
  double? _predictedWeight; // LLM이 예측한 중량 (g)
  FoodItem? _matchedFoodItem;
  List<FoodItem>? _similarFoods;
  double? _adjustedCalories; // 중량 조정된 칼로리
  double? _adjustedProtein; // 중량 조정된 단백질
  double? _adjustedFat; // 중량 조정된 지방
  double? _adjustedCarbs; // 중량 조정된 탄수화물

  @override
  void initState() {
    super.initState();
    _initializeDatabase();
  }

  Future<void> _initializeDatabase() async {
    if (!_foodDb.isLoaded) {
      setState(() => _isProcessing = true);
      try {
        await _foodDb.loadDatabase();
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('음식 데이터베이스 로드 완료: ${_foodDb.allFoods.length}개'),
              backgroundColor: Colors.green,
            ),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('데이터베이스 로드 실패: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      } finally {
        setState(() => _isProcessing = false);
      }
    }
  }

  // 카메라로 사진 촬영
  Future<void> _takePicture() async {
    try {
      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1920,
        maxHeight: 1080,
        imageQuality: 85,
      );

      if (photo != null) {
        final bytes = await photo.readAsBytes();
        setState(() {
          _imageFile = photo;
          _imageBytes = bytes;
          _isProcessing = true;
        });
        await _processImage();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('카메라 오류: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  // 갤러리에서 사진 선택
  Future<void> _pickFromGallery() async {
    try {
      final XFile? photo = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1080,
        imageQuality: 85,
      );

      if (photo != null) {
        final bytes = await photo.readAsBytes();
        setState(() {
          _imageFile = photo;
          _imageBytes = bytes;
          _isProcessing = true;
        });
        await _processImage();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('갤러리 오류: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  // 이미지 처리 및 음식 인식 (중량 예측 포함)
  Future<void> _processImage() async {
    // TODO: 실제 AI 모델 연동 시 여기를 수정
    // 현재는 Mock 데이터로 시뮬레이션

    await Future.delayed(const Duration(seconds: 2));

    // Mock: 랜덤으로 음식 이름과 중량 예측
    final mockFoodData = [
      {'name': '김치찌개', 'weight': 450.0},
      {'name': '된장찌개', 'weight': 400.0},
      {'name': '비빔밥', 'weight': 350.0},
      {'name': '불고기', 'weight': 250.0},
      {'name': '삼겹살', 'weight': 180.0},
      {'name': '김밥', 'weight': 200.0},
      {'name': '라면', 'weight': 550.0},
      {'name': '치킨', 'weight': 300.0},
      {'name': '피자', 'weight': 280.0},
      {'name': '햄버거', 'weight': 220.0},
      {'name': '샐러드', 'weight': 150.0},
      {'name': '스테이크', 'weight': 200.0},
    ];

    final randomIndex = DateTime.now().second % mockFoodData.length;
    final predictedFood = mockFoodData[randomIndex]['name'] as String;
    final predictedWeight = mockFoodData[randomIndex]['weight'] as double;

    setState(() {
      _recognizedFood = predictedFood;
      _predictedWeight = predictedWeight;
      _isProcessing = false;
    });

    // 공공데이터에서 유사한 음식 찾기
    await _searchSimilarFoods(predictedFood);
  }

  // 유사한 음식 검색
  Future<void> _searchSimilarFoods(String foodName) async {
    if (!_foodDb.isLoaded) return;

    final results = _foodDb.findSimilarFoods(foodName, limit: 10);

    setState(() {
      _similarFoods = results;
      _matchedFoodItem = results.isNotEmpty ? results.first : null;
    });

    // 예측된 중량을 기반으로 영양정보 재계산
    if (_matchedFoodItem != null && _predictedWeight != null) {
      _calculateAdjustedNutrition(_matchedFoodItem!, _predictedWeight!);
    }

    if (results.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('일치하는 음식을 찾을 수 없습니다.'),
            backgroundColor: Colors.orange,
          ),
        );
      }
    }
  }

  // 예측된 중량을 기반으로 영양정보 재계산
  void _calculateAdjustedNutrition(FoodItem foodItem, double predictedWeight) {
    // 데이터베이스의 영양정보는 100g 기준이므로 예측된 중량에 맞게 조정
    final weightRatio = predictedWeight / 100.0;

    setState(() {
      _adjustedCalories = foodItem.calories * weightRatio;
      _adjustedProtein = foodItem.protein * weightRatio;
      _adjustedFat = foodItem.fat * weightRatio;
      _adjustedCarbs = foodItem.carbohydrates * weightRatio;
    });
  }

  // 선택한 음식을 데이터베이스에 저장
  Future<void> _saveMeal(FoodItem foodItem) async {
    // 조정된 영양정보 사용 (예측 중량 기반)
    final meal = MealNutrition(
      dateTime: DateTime.now(),
      mealType: _getMealTypeByTime(DateTime.now()),
      foodName: foodItem.name,
      calories: _adjustedCalories ?? foodItem.calories,
      carbohydrates: _adjustedCarbs ?? foodItem.carbohydrates,
      protein: _adjustedProtein ?? foodItem.protein,
      fat: _adjustedFat ?? foodItem.fat,
      notes: '카메라로 촬영한 음식 (인식: $_recognizedFood, 예측 중량: ${_predictedWeight?.toStringAsFixed(0)}g)',
    );

    try {
      await DatabaseHelper.instance.create(meal);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('식단이 저장되었습니다!'),
            backgroundColor: Colors.green,
          ),
        );
        Navigator.pop(context, true); // 저장 후 이전 화면으로 돌아감
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('저장 실패: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  // 현재 시간에 따라 식사 종류 결정
  String _getMealTypeByTime(DateTime time) {
    final hour = time.hour;
    if (hour >= 6 && hour < 10) return '아침';
    if (hour >= 10 && hour < 15) return '점심';
    if (hour >= 15 && hour < 20) return '저녁';
    return '간식';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('음식 촬영 및 인식'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: _isProcessing
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('처리 중...'),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 이미지 표시 영역
                  Card(
                    elevation: 4,
                    child: Container(
                      height: 300,
                      decoration: BoxDecoration(
                        color: Colors.grey[200],
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: _imageFile != null
                          ? ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: kIsWeb
                                  ? Image.memory(
                                      _imageBytes!,
                                      fit: BoxFit.cover,
                                    )
                                  : Image.file(
                                      File(_imageFile!.path),
                                      fit: BoxFit.cover,
                                    ),
                            )
                          : const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.camera_alt,
                                    size: 80,
                                    color: Colors.grey,
                                  ),
                                  SizedBox(height: 16),
                                  Text(
                                    '음식 사진을 촬영해주세요',
                                    style: TextStyle(
                                      fontSize: 16,
                                      color: Colors.grey,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // 카메라/갤러리 버튼
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _takePicture,
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('카메라'),
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.all(16),
                            backgroundColor: Colors.blue,
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _pickFromGallery,
                          icon: const Icon(Icons.photo_library),
                          label: const Text('갤러리'),
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.all(16),
                            backgroundColor: Colors.green,
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),

                  // 인식 결과 표시
                  if (_recognizedFood != null) ...[
                    const SizedBox(height: 24),
                    Card(
                      elevation: 3,
                      color: Colors.blue.shade50,
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(
                                  Icons.lightbulb,
                                  color: Colors.orange,
                                ),
                                const SizedBox(width: 8),
                                const Text(
                                  'AI 인식 결과:',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _recognizedFood!,
                              style: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: Colors.blue,
                              ),
                            ),
                            if (_predictedWeight != null) ...[
                              const SizedBox(height: 4),
                              Text(
                                '예측 중량: ${_predictedWeight!.toStringAsFixed(0)}g',
                                style: TextStyle(
                                  fontSize: 16,
                                  color: Colors.grey[700],
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ],

                  // 매칭된 음식 정보
                  if (_matchedFoodItem != null) ...[
                    const SizedBox(height: 16),
                    const Text(
                      '가장 유사한 음식:',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Card(
                      elevation: 3,
                      color: Colors.green.shade50,
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _matchedFoodItem!.name,
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 12),

                            // 기존 영양정보 (100g 기준)
                            Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: Colors.grey[100],
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    '데이터베이스 영양정보 (100g 기준)',
                                    style: TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.w500,
                                      color: Colors.grey[600],
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    '칼로리: ${_matchedFoodItem!.calories}kcal | '
                                    '탄: ${_matchedFoodItem!.carbohydrates}g, '
                                    '단: ${_matchedFoodItem!.protein}g, '
                                    '지: ${_matchedFoodItem!.fat}g',
                                    style: const TextStyle(fontSize: 14),
                                  ),
                                ],
                              ),
                            ),

                            // 조정된 영양정보 (예측 중량 기준)
                            if (_predictedWeight != null) ...[
                              const SizedBox(height: 12),
                              Container(
                                padding: const EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                  color: Colors.blue[50],
                                  borderRadius: BorderRadius.circular(8),
                                  border: Border.all(color: Colors.blue.shade200),
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      '조정된 영양정보 (${_predictedWeight!.toStringAsFixed(0)}g 기준)',
                                      style: const TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.w500,
                                        color: Colors.blue,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      '칼로리: ${_adjustedCalories?.toStringAsFixed(1)}kcal | '
                                      '탄: ${_adjustedCarbs?.toStringAsFixed(1)}g, '
                                      '단: ${_adjustedProtein?.toStringAsFixed(1)}g, '
                                      '지: ${_adjustedFat?.toStringAsFixed(1)}g',
                                      style: const TextStyle(fontSize: 14),
                                    ),
                                  ],
                                ),
                              ),
                            ],

                            const SizedBox(height: 16),
                            SizedBox(
                              width: double.infinity,
                              child: ElevatedButton(
                                onPressed: () => _saveMeal(_matchedFoodItem!),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.green,
                                  foregroundColor: Colors.white,
                                  padding: const EdgeInsets.all(16),
                                ),
                                child: const Text(
                                  '식단에 저장',
                                  style: TextStyle(fontSize: 16),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],

                  // 다른 유사한 음식 목록
                  if (_similarFoods != null && _similarFoods!.length > 1) ...[
                    const SizedBox(height: 16),
                    const Text(
                      '다른 유사한 음식들:',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    ListView.builder(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      itemCount: _similarFoods!.length - 1,
                      itemBuilder: (context, index) {
                        final food = _similarFoods![index + 1];

                        // 예측 중량을 기준으로 영양정보 계산
                        final weightRatio = _predictedWeight != null ? _predictedWeight! / 100.0 : 1.0;
                        final adjustedCal = food.calories * weightRatio;
                        final adjustedCarbs = food.carbohydrates * weightRatio;
                        final adjustedProtein = food.protein * weightRatio;
                        final adjustedFat = food.fat * weightRatio;

                        return Card(
                          child: Padding(
                            padding: const EdgeInsets.all(12.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  food.name,
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 4),

                                // 기존 영양정보 (100g 기준)
                                Text(
                                  '데이터베이스 (100g): ${food.calories}kcal | '
                                  '탄: ${food.carbohydrates}g, 단: ${food.protein}g, 지: ${food.fat}g',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.grey[600],
                                  ),
                                ),

                                // 조정된 영양정보 (예측 중량 기준)
                                if (_predictedWeight != null) ...[
                                  Text(
                                    '조정됨 (${_predictedWeight!.toStringAsFixed(0)}g): ${adjustedCal.toStringAsFixed(1)}kcal | '
                                    '탄: ${adjustedCarbs.toStringAsFixed(1)}g, 단: ${adjustedProtein.toStringAsFixed(1)}g, 지: ${adjustedFat.toStringAsFixed(1)}g',
                                    style: const TextStyle(
                                      fontSize: 12,
                                      color: Colors.blue,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ],

                                const SizedBox(height: 8),
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.end,
                                  children: [
                                    ElevatedButton(
                                      onPressed: () {
                                        // 선택된 음식으로 조정된 영양정보 계산
                                        if (_predictedWeight != null) {
                                          _calculateAdjustedNutrition(food, _predictedWeight!);
                                        }
                                        _saveMeal(food);
                                      },
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.blue,
                                        foregroundColor: Colors.white,
                                        minimumSize: const Size(80, 36),
                                      ),
                                      child: const Text('선택'),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ],
                ],
              ),
            ),
    );
  }
}
