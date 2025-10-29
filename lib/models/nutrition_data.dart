class NutritionData {
  final DateTime date;
  final double calories;
  final double sodium;
  final double sugar;
  final double carbohydrates;

  NutritionData({
    required this.date,
    required this.calories,
    required this.sodium,
    required this.sugar,
    required this.carbohydrates,
  });

  // 샘플 데이터 생성 (테스트용)
  static List<NutritionData> generateSampleData() {
    final now = DateTime.now();
    final List<NutritionData> data = [];

    // 최근 365일의 샘플 데이터 생성
    for (int i = 0; i < 365; i++) {
      final date = now.subtract(Duration(days: i));
      data.add(NutritionData(
        date: date,
        calories: 1800 + (i % 10) * 50 + (i % 3) * 100, // 1800-2300 kcal
        sodium: 1500 + (i % 15) * 100, // 1500-3000 mg
        sugar: 30 + (i % 8) * 5, // 30-65 g
        carbohydrates: 250 + (i % 12) * 10, // 250-360 g
      ));
    }

    return data;
  }
}
