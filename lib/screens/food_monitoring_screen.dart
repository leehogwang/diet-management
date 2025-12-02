import 'package:flutter/material.dart';
import '../models/food_record.dart';
import 'feedback_analysis_screen.dart';

class FoodMonitoringScreen extends StatefulWidget {
  const FoodMonitoringScreen({super.key});

  @override
  State<FoodMonitoringScreen> createState() => _FoodMonitoringScreenState();
}

class _FoodMonitoringScreenState extends State<FoodMonitoringScreen> {
  late List<FoodRecord> _todayRecords;
  late DailyNutritionSummary _summary;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  void _loadData() {
    _todayRecords = DummyFoodData.getTodayRecords();
    _summary = DailyNutritionSummary.fromRecords(_todayRecords);
  }

  Color _getProgressColor(double percentage) {
    if (percentage > 100) return Colors.red;
    if (percentage > 80) return Colors.orange;
    return Colors.green;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('음식 모니터링'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 날짜 표시
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    const Icon(Icons.calendar_today, size: 20),
                    const SizedBox(width: 8),
                    Text(
                      '오늘 (${DateTime.now().month}/${DateTime.now().day})',
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // 영양 요약 카드
            Card(
              color: Colors.blue.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '오늘의 영양 요약',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildNutritionProgress(
                      '칼로리',
                      _summary.totalCalories.toDouble(),
                      DailyNutritionSummary.recommendedCalories.toDouble(),
                      'kcal',
                    ),
                    _buildNutritionProgress(
                      '단백질',
                      _summary.totalProtein,
                      DailyNutritionSummary.recommendedProtein,
                      'g',
                    ),
                    _buildNutritionProgress(
                      '탄수화물',
                      _summary.totalCarbs,
                      DailyNutritionSummary.recommendedCarbs,
                      'g',
                    ),
                    _buildNutritionProgress(
                      '지방',
                      _summary.totalFat,
                      DailyNutritionSummary.recommendedFat,
                      'g',
                    ),
                    _buildNutritionProgress(
                      '나트륨',
                      _summary.totalSodium,
                      DailyNutritionSummary.recommendedSodium,
                      'mg',
                    ),
                    _buildNutritionProgress(
                      '당',
                      _summary.totalSugar,
                      DailyNutritionSummary.recommendedSugar,
                      'g',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // AI 피드백 버튼
            ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => FeedbackAnalysisScreen(
                      records: _todayRecords,
                      summary: _summary,
                    ),
                  ),
                );
              },
              icon: const Icon(Icons.auto_awesome),
              label: const Text('AI 영양 피드백 받기'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.purple,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.all(16),
                textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 24),

            // 식사별 기록
            const Text(
              '오늘의 식사 기록',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ..._buildMealSections(),
          ],
        ),
      ),
    );
  }

  Widget _buildNutritionProgress(
    String label,
    double current,
    double recommended,
    String unit,
  ) {
    final percentage = (current / recommended) * 100;
    final color = _getProgressColor(percentage);

    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: const TextStyle(fontWeight: FontWeight.w500),
              ),
              Text(
                '${current.toStringAsFixed(1)} / ${recommended.toStringAsFixed(0)} $unit',
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: (percentage / 100).clamp(0.0, 1.0),
              backgroundColor: Colors.grey.shade300,
              color: color,
              minHeight: 8,
            ),
          ),
          if (percentage > 100)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                '${(percentage - 100).toStringAsFixed(0)}% 초과',
                style: TextStyle(
                  fontSize: 12,
                  color: color,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
        ],
      ),
    );
  }

  List<Widget> _buildMealSections() {
    final mealTypeOrder = ['breakfast', 'lunch', 'dinner', 'snack'];
    final mealTypeLabels = {
      'breakfast': '아침',
      'lunch': '점심',
      'dinner': '저녁',
      'snack': '간식',
    };
    final mealTypeIcons = {
      'breakfast': Icons.wb_sunny,
      'lunch': Icons.wb_cloudy,
      'dinner': Icons.nightlight_round,
      'snack': Icons.cookie,
    };

    final sections = <Widget>[];

    for (final mealType in mealTypeOrder) {
      final meals = _todayRecords.where((r) => r.mealType == mealType).toList();
      if (meals.isEmpty) continue;

      sections.add(
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(mealTypeIcons[mealType], size: 24),
                    const SizedBox(width: 8),
                    Text(
                      mealTypeLabels[mealType]!,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const Divider(),
                ...meals.map((meal) => _buildFoodItem(meal)),
              ],
            ),
          ),
        ),
      );
      sections.add(const SizedBox(height: 12));
    }

    return sections;
  }

  Widget _buildFoodItem(FoodRecord record) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  record.foodName,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${record.calories}kcal | 단백질 ${record.protein}g | 나트륨 ${record.sodium}mg',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey.shade700,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.blue.shade100,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              '${record.calories}kcal',
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
