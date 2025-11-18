import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/meal_nutrition.dart';
import '../database/database_helper.dart';
import '../services/food_nutrition_api.dart';

class MealCorrectionScreen extends StatefulWidget {
  const MealCorrectionScreen({super.key});

  @override
  State<MealCorrectionScreen> createState() => _MealCorrectionScreenState();
}

class _MealCorrectionScreenState extends State<MealCorrectionScreen> {
  List<MealNutrition> _meals = [];
  bool _isLoading = false;
  String? _apiKey;

  @override
  void initState() {
    super.initState();
    _loadMeals();
  }

  Future<void> _loadMeals() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final meals = await DatabaseHelper.instance.readAll();
      setState(() {
        _meals = meals;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('데이터 로드 실패: $e')),
        );
      }
    }
  }

  Future<void> _showApiKeyDialog() async {
    final controller = TextEditingController(text: _apiKey ?? '');

    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('API 키 설정'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '식품의약품안전처 공공데이터포털에서\n발급받은 API 키를 입력하세요.',
              style: TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                hintText: 'API 키 입력',
                border: OutlineInputBorder(),
              ),
              maxLines: 3,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, controller.text),
            child: const Text('저장'),
          ),
        ],
      ),
    );

    if (result != null && result.isNotEmpty) {
      setState(() {
        _apiKey = result;
      });
      FoodNutritionAPI.setApiKey(result);

      // API 키 검증
      final isValid = await FoodNutritionAPI.validateApiKey();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(isValid ? 'API 키가 설정되었습니다!' : 'API 키가 유효하지 않습니다.'),
            backgroundColor: isValid ? Colors.green : Colors.orange,
          ),
        );
      }
    }
  }

  Future<void> _correctMealNutrition(MealNutrition meal) async {
    if (_apiKey == null || _apiKey!.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('먼저 API 키를 설정해주세요.'),
          backgroundColor: Colors.orange,
        ),
      );
      _showApiKeyDialog();
      return;
    }

    // 로딩 다이얼로그 표시
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(
        child: CircularProgressIndicator(),
      ),
    );

    try {
      // API로 식품 검색
      final results = await FoodNutritionAPI.searchFood(meal.foodName);

      if (mounted) {
        Navigator.pop(context); // 로딩 다이얼로그 닫기

        if (results.isEmpty) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('검색 결과가 없습니다.'),
              backgroundColor: Colors.orange,
            ),
          );
          return;
        }

        // 검색 결과 선택 다이얼로그
        final selected = await showDialog<FoodNutritionInfo>(
          context: context,
          builder: (context) => AlertDialog(
            title: Text('\'${meal.foodName}\' 검색 결과'),
            content: SizedBox(
              width: double.maxFinite,
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: results.length,
                itemBuilder: (context, index) {
                  final item = results[index];
                  return Card(
                    child: ListTile(
                      title: Text(item.foodName),
                      subtitle: Text(
                        '칼로리: ${item.calories}kcal\n'
                        '탄수화물: ${item.carbohydrates}g, '
                        '단백질: ${item.protein}g, '
                        '지방: ${item.fat}g',
                        style: const TextStyle(fontSize: 12),
                      ),
                      trailing: const Icon(Icons.arrow_forward),
                      onTap: () => Navigator.pop(context, item),
                    ),
                  );
                },
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('취소'),
              ),
            ],
          ),
        );

        // 선택된 결과로 데이터 업데이트
        if (selected != null) {
          final updatedMeal = meal.copyWith(
            calories: selected.calories,
            carbohydrates: selected.carbohydrates,
            protein: selected.protein,
            fat: selected.fat,
          );

          await DatabaseHelper.instance.update(updatedMeal);
          await _loadMeals(); // 목록 새로고침

          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('영양 정보가 수정되었습니다!'),
                backgroundColor: Colors.green,
              ),
            );
          }
        }
      }
    } catch (e) {
      if (mounted) {
        Navigator.pop(context); // 로딩 다이얼로그 닫기
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('오류 발생: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _deleteMeal(MealNutrition meal) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('삭제 확인'),
        content: Text('\'${meal.foodName}\' 데이터를 삭제하시겠습니까?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('취소'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('삭제'),
          ),
        ],
      ),
    );

    if (confirm == true && meal.id != null) {
      await DatabaseHelper.instance.delete(meal.id!);
      await _loadMeals();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('데이터가 삭제되었습니다.'),
            backgroundColor: Colors.green,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('식단 데이터 후보정'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.key),
            tooltip: 'API 키 설정',
            onPressed: _showApiKeyDialog,
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: '새로고침',
            onPressed: _loadMeals,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _meals.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.inbox,
                        size: 100,
                        color: Colors.grey[400],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        '저장된 식단 데이터가 없습니다.',
                        style: TextStyle(
                          fontSize: 18,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _meals.length,
                  itemBuilder: (context, index) {
                    final meal = _meals[index];
                    return Card(
                      elevation: 3,
                      margin: const EdgeInsets.only(bottom: 12),
                      child: ExpansionTile(
                        leading: CircleAvatar(
                          backgroundColor: _getMealTypeColor(meal.mealType),
                          child: Text(
                            meal.mealType[0],
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                        title: Text(
                          meal.foodName,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        subtitle: Text(
                          '${DateFormat('yyyy-MM-dd HH:mm').format(meal.dateTime)} | ${meal.mealType}',
                          style: const TextStyle(fontSize: 12),
                        ),
                        children: [
                          Padding(
                            padding: const EdgeInsets.all(16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                _buildNutritionRow(
                                    '칼로리', '${meal.calories} kcal'),
                                _buildNutritionRow(
                                    '탄수화물', '${meal.carbohydrates} g'),
                                _buildNutritionRow(
                                    '단백질', '${meal.protein} g'),
                                _buildNutritionRow('지방', '${meal.fat} g'),
                                if (meal.notes != null &&
                                    meal.notes!.isNotEmpty) ...[
                                  const Divider(),
                                  Text(
                                    '메모: ${meal.notes}',
                                    style: const TextStyle(
                                      fontSize: 12,
                                      fontStyle: FontStyle.italic,
                                      color: Colors.grey,
                                    ),
                                  ),
                                ],
                                const SizedBox(height: 12),
                                Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceEvenly,
                                  children: [
                                    ElevatedButton.icon(
                                      onPressed: () =>
                                          _correctMealNutrition(meal),
                                      icon: const Icon(Icons.search, size: 18),
                                      label: const Text('보정'),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.blue.shade600,
                                        foregroundColor: Colors.white,
                                      ),
                                    ),
                                    OutlinedButton.icon(
                                      onPressed: () => _deleteMeal(meal),
                                      icon: const Icon(Icons.delete, size: 18),
                                      label: const Text('삭제'),
                                      style: OutlinedButton.styleFrom(
                                        foregroundColor: Colors.red,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
      floatingActionButton: _apiKey == null || _apiKey!.isEmpty
          ? FloatingActionButton.extended(
              onPressed: _showApiKeyDialog,
              icon: const Icon(Icons.key),
              label: const Text('API 키 설정'),
              backgroundColor: Colors.orange,
            )
          : null,
    );
  }

  Widget _buildNutritionRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.black87,
            ),
          ),
        ],
      ),
    );
  }

  Color _getMealTypeColor(String mealType) {
    switch (mealType) {
      case '아침':
        return Colors.orange;
      case '점심':
        return Colors.green;
      case '저녁':
        return Colors.blue;
      case '간식':
        return Colors.purple;
      default:
        return Colors.grey;
    }
  }
}
