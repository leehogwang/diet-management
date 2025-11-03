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
  TimePeriod _selectedPeriod = TimePeriod.week;
  final List<NutritionData> _allData = NutritionData.generateSampleData();

  List<NutritionData> get _filteredData {
    final now = DateTime.now();
    switch (_selectedPeriod) {
      case TimePeriod.day:
        // 오늘 하루의 시간대별 데이터 (24시간)
        return _allData.where((data) {
          return data.date.year == now.year &&
              data.date.month == now.month &&
              data.date.day == now.day;
        }).toList();
      case TimePeriod.week:
        // 최근 7일
        return _allData
            .where((data) =>
                data.date.isAfter(now.subtract(const Duration(days: 7))))
            .toList()
          ..sort((a, b) => a.date.compareTo(b.date));
      case TimePeriod.month:
        // 최근 30일
        return _allData
            .where((data) =>
                data.date.isAfter(now.subtract(const Duration(days: 30))))
            .toList()
          ..sort((a, b) => a.date.compareTo(b.date));
      case TimePeriod.year:
        // 최근 12개월
        return _allData
            .where((data) =>
                data.date.isAfter(now.subtract(const Duration(days: 365))))
            .toList()
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
      ),
      body: SingleChildScrollView(
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
        alignment: BarChartAlignment.spaceAround,
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