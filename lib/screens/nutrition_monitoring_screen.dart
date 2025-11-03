import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/nutrition_data.dart';

enum TimePeriod { day, week, month, year }

class NutritionMonitoringScreen extends StatefulWidget {
  const NutritionMonitoringScreen({super.key});

  @override
  State<NutritionMonitoringScreen> createState() =>
      _NutritionMonitoringScreenState();
}

class _NutritionMonitoringScreenState extends State<NutritionMonitoringScreen> {
  TimePeriod _selectedPeriod = TimePeriod.day;
  List<NutritionData> _allData = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadNutritionData();
  }

  Future<void> _loadNutritionData() async {
    final data = await NutritionData.loadRealData();
    if (mounted) {
      setState(() {
        _allData = data;
        _isLoading = false;
      });
    }
  }

  List<NutritionData> get _filteredData {
    final now = DateTime.now();
    switch (_selectedPeriod) {
      case TimePeriod.day:
        // 오늘 하루의 시간대별 데이터 (24시간) - 같은 시간대 합치기
        final todayData = _allData.where((data) {
          return data.date.year == now.year &&
              data.date.month == now.month &&
              data.date.day == now.day;
        }).toList();

        // 시간대별로 그룹화하여 합산
        final Map<int, NutritionData> hourlyData = {};
        for (final data in todayData) {
          final hour = data.date.hour;
          if (hourlyData.containsKey(hour)) {
            // 같은 시간대 데이터가 있으면 합산
            final existing = hourlyData[hour]!;
            hourlyData[hour] = NutritionData(
              date: DateTime(now.year, now.month, now.day, hour),
              calories: existing.calories + data.calories,
              sodium: existing.sodium + data.sodium,
              sugar: existing.sugar + data.sugar,
              carbohydrates: existing.carbohydrates + data.carbohydrates,
              imagePath: existing.imagePath, // 첫 번째 이미지 경로 유지
            );
          } else {
            // 새로운 시간대
            hourlyData[hour] = NutritionData(
              date: DateTime(now.year, now.month, now.day, hour),
              calories: data.calories,
              sodium: data.sodium,
              sugar: data.sugar,
              carbohydrates: data.carbohydrates,
              imagePath: data.imagePath,
            );
          }
        }

        // 시간 순서대로 정렬하여 반환
        final result = hourlyData.values.toList()
          ..sort((a, b) => a.date.compareTo(b.date));
        return result;

      case TimePeriod.week:
        // 최근 7일 - 같은 날짜 합치기
        final weekData = _allData
            .where((data) =>
                data.date.isAfter(now.subtract(const Duration(days: 7))))
            .toList();

        // 날짜별로 그룹화하여 합산
        final Map<String, NutritionData> dailyData = {};
        for (final data in weekData) {
          final dateKey = '${data.date.year}-${data.date.month}-${data.date.day}';
          if (dailyData.containsKey(dateKey)) {
            // 같은 날짜 데이터가 있으면 합산
            final existing = dailyData[dateKey]!;
            dailyData[dateKey] = NutritionData(
              date: DateTime(data.date.year, data.date.month, data.date.day),
              calories: existing.calories + data.calories,
              sodium: existing.sodium + data.sodium,
              sugar: existing.sugar + data.sugar,
              carbohydrates: existing.carbohydrates + data.carbohydrates,
              imagePath: existing.imagePath,
            );
          } else {
            // 새로운 날짜
            dailyData[dateKey] = NutritionData(
              date: DateTime(data.date.year, data.date.month, data.date.day),
              calories: data.calories,
              sodium: data.sodium,
              sugar: data.sugar,
              carbohydrates: data.carbohydrates,
              imagePath: data.imagePath,
            );
          }
        }

        return dailyData.values.toList()
          ..sort((a, b) => a.date.compareTo(b.date));

      case TimePeriod.month:
        // 최근 30일 - 같은 날짜 합치기
        final monthData = _allData
            .where((data) =>
                data.date.isAfter(now.subtract(const Duration(days: 30))))
            .toList();

        // 날짜별로 그룹화하여 합산
        final Map<String, NutritionData> dailyData = {};
        for (final data in monthData) {
          final dateKey = '${data.date.year}-${data.date.month}-${data.date.day}';
          if (dailyData.containsKey(dateKey)) {
            // 같은 날짜 데이터가 있으면 합산
            final existing = dailyData[dateKey]!;
            dailyData[dateKey] = NutritionData(
              date: DateTime(data.date.year, data.date.month, data.date.day),
              calories: existing.calories + data.calories,
              sodium: existing.sodium + data.sodium,
              sugar: existing.sugar + data.sugar,
              carbohydrates: existing.carbohydrates + data.carbohydrates,
              imagePath: existing.imagePath,
            );
          } else {
            // 새로운 날짜
            dailyData[dateKey] = NutritionData(
              date: DateTime(data.date.year, data.date.month, data.date.day),
              calories: data.calories,
              sodium: data.sodium,
              sugar: data.sugar,
              carbohydrates: data.carbohydrates,
              imagePath: data.imagePath,
            );
          }
        }

        return dailyData.values.toList()
          ..sort((a, b) => a.date.compareTo(b.date));

      case TimePeriod.year:
        // 최근 12개월 - 같은 날짜 합치기
        final yearData = _allData
            .where((data) =>
                data.date.isAfter(now.subtract(const Duration(days: 365))))
            .toList();

        // 날짜별로 그룹화하여 합산
        final Map<String, NutritionData> dailyData = {};
        for (final data in yearData) {
          final dateKey = '${data.date.year}-${data.date.month}-${data.date.day}';
          if (dailyData.containsKey(dateKey)) {
            // 같은 날짜 데이터가 있으면 합산
            final existing = dailyData[dateKey]!;
            dailyData[dateKey] = NutritionData(
              date: DateTime(data.date.year, data.date.month, data.date.day),
              calories: existing.calories + data.calories,
              sodium: existing.sodium + data.sodium,
              sugar: existing.sugar + data.sugar,
              carbohydrates: existing.carbohydrates + data.carbohydrates,
              imagePath: existing.imagePath,
            );
          } else {
            // 새로운 날짜
            dailyData[dateKey] = NutritionData(
              date: DateTime(data.date.year, data.date.month, data.date.day),
              calories: data.calories,
              sodium: data.sodium,
              sugar: data.sugar,
              carbohydrates: data.carbohydrates,
              imagePath: data.imagePath,
            );
          }
        }

        return dailyData.values.toList()
          ..sort((a, b) => a.date.compareTo(b.date));
    }
  }

  String _getPeriodLabel(TimePeriod period) {
    switch (period) {
      case TimePeriod.day:
        return '1일';
      case TimePeriod.week:
        return '일주일';
      case TimePeriod.month:
        return '한 달';
      case TimePeriod.year:
        return '1년';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('영양정보 모니터링'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadNutritionData,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _allData.isEmpty
              ? const Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.no_food, size: 80, color: Colors.grey),
                      SizedBox(height: 16),
                      Text(
                        '분석된 음식 데이터가 없습니다\n사진을 찍어 음식을 분석해주세요',
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                )
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // 기간 선택 버튼
                      _buildPeriodSelector(),
                      const SizedBox(height: 24),

                      // 칼로리 막대 그래프
                      const Text(
                        '칼로리 (kcal)',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Card(
                        elevation: 4,
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: SizedBox(
                            height: 250,
                            child: _buildCaloriesBarChart(),
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),

                      // 영양소 선 그래프
                      const Text(
                        '영양소',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Card(
                        elevation: 4,
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: SizedBox(
                            height: 250,
                            child: _buildNutrientsLineChart(),
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),

                      // 범례
                      _buildLegend(),
                    ],
                  ),
                ),
    );
  }

  Widget _buildPeriodSelector() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: TimePeriod.values.map((period) {
        final isSelected = _selectedPeriod == period;
        return Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: ElevatedButton(
              onPressed: () {
                setState(() {
                  _selectedPeriod = period;
                });
              },
              style: ElevatedButton.styleFrom(
                backgroundColor:
                    isSelected ? Theme.of(context).colorScheme.primary : null,
                foregroundColor: isSelected ? Colors.white : null,
              ),
              child: Text(_getPeriodLabel(period)),
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildCaloriesBarChart() {
    final data = _filteredData;
    if (data.isEmpty) {
      return const Center(child: Text('데이터가 없습니다'));
    }

    return BarChart(
      BarChartData(
        alignment: BarChartAlignment.start,
        maxY: 2500,
        minY: 0,
        barTouchData: BarTouchData(
          touchTooltipData: BarTouchTooltipData(
            getTooltipItem: (group, groupIndex, rod, rodIndex) {
              return BarTooltipItem(
                '${rod.toY.toInt()} kcal',
                const TextStyle(color: Colors.white),
              );
            },
          ),
        ),
        titlesData: FlTitlesData(
          show: true,
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index < 0 || index >= data.length) return const Text('');

                final date = data[index].date;
                String label = '';

                switch (_selectedPeriod) {
                  case TimePeriod.day:
                    label = '${date.hour}시';
                    break;
                  case TimePeriod.week:
                    label = '${date.month}/${date.day}';
                    break;
                  case TimePeriod.month:
                    if (index % 5 == 0) {
                      label = '${date.month}/${date.day}';
                    }
                    break;
                  case TimePeriod.year:
                    if (index % 30 == 0) {
                      label = '${date.month}월';
                    }
                    break;
                }

                return Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    label,
                    style: const TextStyle(fontSize: 10),
                  ),
                );
              },
              reservedSize: 30,
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 40,
              getTitlesWidget: (value, meta) {
                return Text(
                  '${value.toInt()}',
                  style: const TextStyle(fontSize: 10),
                );
              },
            ),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
        ),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 500,
        ),
        borderData: FlBorderData(show: false),
        barGroups: List.generate(data.length, (index) {
          return BarChartGroupData(
            x: index,
            barRods: [
              BarChartRodData(
                toY: data[index].calories,
                color: Colors.blue,
                width: _selectedPeriod == TimePeriod.year ? 3 : 16,
                borderRadius: BorderRadius.circular(4),
              ),
            ],
          );
        }),
      ),
    );
  }

  Widget _buildNutrientsLineChart() {
    final data = _filteredData;
    if (data.isEmpty) {
      return const Center(child: Text('데이터가 없습니다'));
    }

    return LineChart(
      LineChartData(
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipItems: (touchedSpots) {
              return touchedSpots.map((spot) {
                final dataPoint = data[spot.x.toInt()];
                String label = '';
                double value = 0;

                if (spot.barIndex == 0) {
                  label = '당류';
                  value = dataPoint.sugar;
                } else if (spot.barIndex == 1) {
                  label = '나트륨';
                  value = dataPoint.sodium;
                } else {
                  label = '탄수화물';
                  value = dataPoint.carbohydrates;
                }

                return LineTooltipItem(
                  '$label\n${value.toInt()}${spot.barIndex == 1 ? "mg" : "g"}',
                  const TextStyle(color: Colors.white),
                );
              }).toList();
            },
          ),
        ),
        titlesData: FlTitlesData(
          show: true,
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index < 0 || index >= data.length) return const Text('');

                final date = data[index].date;
                String label = '';

                switch (_selectedPeriod) {
                  case TimePeriod.day:
                    label = '${date.hour}시';
                    break;
                  case TimePeriod.week:
                    label = '${date.month}/${date.day}';
                    break;
                  case TimePeriod.month:
                    if (index % 5 == 0) {
                      label = '${date.month}/${date.day}';
                    }
                    break;
                  case TimePeriod.year:
                    if (index % 30 == 0) {
                      label = '${date.month}월';
                    }
                    break;
                }

                return Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    label,
                    style: const TextStyle(fontSize: 10),
                  ),
                );
              },
              reservedSize: 30,
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 50,
              getTitlesWidget: (value, meta) {
                return Text(
                  '${value.toInt()}',
                  style: const TextStyle(fontSize: 10),
                );
              },
            ),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
        ),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
        ),
        borderData: FlBorderData(show: false),
        lineBarsData: [
          // 당류 (Sugar)
          LineChartBarData(
            spots: List.generate(data.length, (index) {
              return FlSpot(index.toDouble(), data[index].sugar);
            }),
            isCurved: true,
            color: Colors.red,
            barWidth: 2,
            dotData: FlDotData(
              show: _selectedPeriod == TimePeriod.week ||
                  _selectedPeriod == TimePeriod.day,
            ),
          ),
          // 나트륨 (Sodium) - 스케일 조정을 위해 10으로 나눔
          LineChartBarData(
            spots: List.generate(data.length, (index) {
              return FlSpot(index.toDouble(), data[index].sodium / 10);
            }),
            isCurved: true,
            color: Colors.orange,
            barWidth: 2,
            dotData: FlDotData(
              show: _selectedPeriod == TimePeriod.week ||
                  _selectedPeriod == TimePeriod.day,
            ),
          ),
          // 탄수화물 (Carbohydrates)
          LineChartBarData(
            spots: List.generate(data.length, (index) {
              return FlSpot(index.toDouble(), data[index].carbohydrates);
            }),
            isCurved: true,
            color: Colors.green,
            barWidth: 2,
            dotData: FlDotData(
              show: _selectedPeriod == TimePeriod.week ||
                  _selectedPeriod == TimePeriod.day,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLegend() {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '범례',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildLegendItem('당류', Colors.red, 'g'),
                _buildLegendItem('나트륨', Colors.orange, 'mg (÷10)'),
                _buildLegendItem('탄수화물', Colors.green, 'g'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegendItem(String label, Color color, String unit) {
    return Row(
      children: [
        Container(
          width: 20,
          height: 3,
          color: color,
        ),
        const SizedBox(width: 8),
        Text(
          '$label ($unit)',
          style: const TextStyle(fontSize: 12),
        ),
      ],
    );
  }
}