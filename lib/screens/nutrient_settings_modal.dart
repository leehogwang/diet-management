import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../theme/app_colors.dart';

class NutrientSettingsModal extends StatefulWidget {
  final String palette;
  final bool isDarkMode;
  final String ringMetric;
  final List<String> barMetrics;

  const NutrientSettingsModal({
    super.key,
    required this.palette,
    required this.isDarkMode,
    required this.ringMetric,
    required this.barMetrics,
  });

  @override
  State<NutrientSettingsModal> createState() => _NutrientSettingsModalState();
}

class _NutrientSettingsModalState extends State<NutrientSettingsModal> {
  late String _ringMetric;
  late List<String> _barMetrics;

  final Map<String, NutrientMeta> _nutrientMeta = {
    'calories': NutrientMeta(label: '칼로리', unit: 'kcal'),
    'carbs': NutrientMeta(label: '탄수화물', unit: 'g'),
    'protein': NutrientMeta(label: '단백질', unit: 'g'),
    'fat': NutrientMeta(label: '지방', unit: 'g'),
    'sugar': NutrientMeta(label: '당류', unit: 'g'),
    'sodium': NutrientMeta(label: '나트륨', unit: 'mg'),
    'fiber': NutrientMeta(label: '식이섬유', unit: 'g'),
  };

  @override
  void initState() {
    super.initState();
    _ringMetric = widget.ringMetric;
    _barMetrics = List.from(widget.barMetrics);
  }

  void _toggleBarMetric(String metric) {
    setState(() {
      if (_barMetrics.contains(metric)) {
        if (_barMetrics.length > 1) {
          _barMetrics.remove(metric);
        }
      } else {
        if (_barMetrics.length < 3) {
          _barMetrics.add(metric);
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = AppTheme.getTheme(widget.palette, widget.isDarkMode);
    final cardColor = theme.colorScheme.surface;
    final textColor = theme.colorScheme.onSurface;
    final textGray = theme.colorScheme.onSurface.withOpacity(0.6);
    final primaryColor = theme.colorScheme.primary;

    return Dialog(
      backgroundColor: Colors.transparent,
      child: Center(
        child: Container(
          width: MediaQuery.of(context).size.width * 0.9,
          constraints: const BoxConstraints(maxWidth: 400),
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(32),
          ),
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    '영양소 표시 설정',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: textColor,
                    ),
                  ),
                  GestureDetector(
                    onTap: () => Navigator.of(context).pop(),
                    child: Icon(Icons.close, color: textGray),
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // Content
              ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 400),
                child: SingleChildScrollView(
                  child: Column(
                    children: [
                      // Ring Chart Metric
                      _buildRingSection(theme, textColor, textGray, primaryColor),

                      const SizedBox(height: 24),

                      // Bar Metrics
                      _buildBarSection(theme, textColor, textGray, primaryColor),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // Save Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).pop({
                      'ringMetric': _ringMetric,
                      'barMetrics': _barMetrics,
                    });
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: primaryColor,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  child: const Text('저장'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRingSection(ThemeData theme, Color textColor, Color textGray, Color primaryColor) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Icon(Icons.adjust_rounded, size: 16, color: textGray),
                const SizedBox(width: 8),
                Text(
                  '메인 목표 (링 차트)',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: textGray,
                    letterSpacing: 1.0,
                  ),
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 12),

        // Grid
        GridView.count(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisCount: 2,
          mainAxisSpacing: 8,
          crossAxisSpacing: 8,
          childAspectRatio: 2.5,
          children: _nutrientMeta.entries.map((entry) {
            final key = entry.key;
            final meta = entry.value;
            final isSelected = _ringMetric == key;

            return GestureDetector(
              onTap: () {
                setState(() {
                  _ringMetric = key;
                });
              },
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: isSelected
                      ? primaryColor.withOpacity(0.15)
                      : (widget.isDarkMode
                          ? Colors.white.withOpacity(0.05)
                          : Colors.grey[100]),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: isSelected ? primaryColor : Colors.transparent,
                    width: 2,
                  ),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      meta.unit,
                      style: TextStyle(
                        fontSize: 10,
                        color: textGray,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      meta.label,
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                        color: isSelected ? primaryColor : textColor,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  Widget _buildBarSection(ThemeData theme, Color textColor, Color textGray, Color primaryColor) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Icon(Icons.bar_chart_rounded, size: 16, color: textGray),
                const SizedBox(width: 8),
                Text(
                  '우선 영양소 (최대 3개)',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: textGray,
                    letterSpacing: 1.0,
                  ),
                ),
              ],
            ),
            Text(
              '${_barMetrics.length}/3',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: primaryColor,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),

        // List
        ..._nutrientMeta.entries.map((entry) {
          final key = entry.key;
          final meta = entry.value;
          final isSelected = _barMetrics.contains(key);
          final isDisabled = !isSelected && _barMetrics.length >= 3;
          if (key == _ringMetric) return const SizedBox.shrink();

          return GestureDetector(
            onTap: isDisabled ? null : () => _toggleBarMetric(key),
            child: Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: isSelected
                    ? primaryColor.withOpacity(0.15)
                    : (widget.isDarkMode
                        ? Colors.white.withOpacity(0.05)
                        : Colors.grey[100]),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: isSelected ? primaryColor : Colors.transparent,
                  width: 2,
                ),
              ),
              child: Row(
                children: [
                  Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: isSelected ? primaryColor : Colors.grey[400],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      meta.label,
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                        color: isSelected
                            ? primaryColor
                            : (isDisabled
                                ? Colors.grey.shade400
                                : textColor),
                      ),
                    ),
                  ),
                  Text(
                    meta.unit,
                    style: TextStyle(
                      fontSize: 12,
                      color: textGray.withOpacity(0.7),
                    ),
                  ),
                ],
              ),
            ),
          );
        }),
      ],
    );
  }
}

class NutrientMeta {
  final String label;
  final String unit;

  NutrientMeta({required this.label, required this.unit});
}
