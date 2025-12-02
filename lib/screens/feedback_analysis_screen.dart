import 'package:flutter/material.dart';
import '../models/food_record.dart';
import '../services/nutrition_feedback_service.dart';

class FeedbackAnalysisScreen extends StatefulWidget {
  final List<FoodRecord> records;
  final DailyNutritionSummary summary;

  const FeedbackAnalysisScreen({
    super.key,
    required this.records,
    required this.summary,
  });

  @override
  State<FeedbackAnalysisScreen> createState() => _FeedbackAnalysisScreenState();
}

class _FeedbackAnalysisScreenState extends State<FeedbackAnalysisScreen> {
  String? _feedback;
  bool _isLoading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _generateFeedback();
  }

  Future<void> _generateFeedback() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final feedback = await NutritionFeedbackService.generateFeedback(
        widget.records,
        widget.summary,
      );

      setState(() {
        _feedback = feedback;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = '피드백 생성 중 오류가 발생했습니다: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI 영양 피드백'),
        backgroundColor: Colors.purple.shade400,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 헤더 카드
            Card(
              color: Colors.purple.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    Icon(Icons.auto_awesome, size: 32, color: Colors.purple.shade700),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'AI 영양 분석',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.purple.shade900,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '오늘의 식단을 분석하고 있습니다',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.purple.shade700,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // 피드백 내용
            if (_isLoading)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(32.0),
                  child: Column(
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 16),
                      Text(
                        'AI가 식단을 분석하고 있습니다...',
                        style: TextStyle(fontSize: 16),
                      ),
                    ],
                  ),
                ),
              )
            else if (_error != null)
              Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Icon(Icons.error_outline, size: 48, color: Colors.red.shade700),
                      const SizedBox(height: 12),
                      Text(
                        _error!,
                        style: TextStyle(color: Colors.red.shade900),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 16),
                      ElevatedButton.icon(
                        onPressed: _generateFeedback,
                        icon: const Icon(Icons.refresh),
                        label: const Text('다시 시도'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.red.shade700,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else if (_feedback != null)
              Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Card(
                    elevation: 4,
                    child: Padding(
                      padding: const EdgeInsets.all(20.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(Icons.lightbulb, color: Colors.amber.shade700, size: 28),
                              const SizedBox(width: 8),
                              const Text(
                                '맞춤 피드백',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                          const Divider(height: 24),
                          Text(
                            _feedback!,
                            style: const TextStyle(
                              fontSize: 16,
                              height: 1.6,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // 영양소별 팁 섹션
                  if (widget.summary.isSodiumExceeded())
                    _buildTipCard(
                      '나트륨 관리 팁',
                      Icons.water_drop,
                      Colors.blue,
                      NutritionFeedbackService.getTipForNutrient('sodium'),
                    ),
                  if (widget.summary.isSugarExceeded())
                    _buildTipCard(
                      '당 관리 팁',
                      Icons.cake,
                      Colors.pink,
                      NutritionFeedbackService.getTipForNutrient('sugar'),
                    ),
                  if (widget.summary.isCaloriesExceeded())
                    _buildTipCard(
                      '칼로리 관리 팁',
                      Icons.local_fire_department,
                      Colors.orange,
                      NutritionFeedbackService.getTipForNutrient('calories'),
                    ),
                  if (widget.summary.totalProtein < DailyNutritionSummary.recommendedProtein * 0.7)
                    _buildTipCard(
                      '단백질 보충 팁',
                      Icons.fitness_center,
                      Colors.green,
                      NutritionFeedbackService.getTipForNutrient('protein'),
                    ),
                ],
              ),

            const SizedBox(height: 24),

            // 재생성 버튼
            if (!_isLoading && _feedback != null)
              OutlinedButton.icon(
                onPressed: _generateFeedback,
                icon: const Icon(Icons.refresh),
                label: const Text('피드백 새로고침'),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                  side: BorderSide(color: Colors.purple.shade400, width: 2),
                  foregroundColor: Colors.purple.shade700,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTipCard(String title, IconData icon, MaterialColor color, String content) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Card(
        color: color.shade50,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(icon, color: color.shade700, size: 24),
                  const SizedBox(width: 8),
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: color.shade900,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                content,
                style: TextStyle(
                  fontSize: 14,
                  height: 1.5,
                  color: color.shade900,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
