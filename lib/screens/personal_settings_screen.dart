import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class PersonalSettingsScreen extends StatefulWidget {
  final String palette;
  final bool isDarkMode;

  const PersonalSettingsScreen({
    super.key,
    required this.palette,
    required this.isDarkMode,
  });

  @override
  State<PersonalSettingsScreen> createState() => _PersonalSettingsScreenState();
}

class _PersonalSettingsScreenState extends State<PersonalSettingsScreen> {
  int? _selectedGender;
  final Map<String, bool> _conditions = {
    'diabetes': false,
    'hypertension': false,
    'kidney': false,
    'allergy': false,
    'none': false,
  };

  void _handleConditionToggle(String key) {
    setState(() {
      if (key == 'none') {
        _conditions.updateAll((k, v) => false);
        _conditions['none'] = !_conditions['none']!;
      } else {
        _conditions[key] = !_conditions[key]!;
        _conditions['none'] = false;
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

    return Theme(
      data: theme,
      child: Scaffold(
        backgroundColor: theme.scaffoldBackgroundColor,
        body: SafeArea(
          child: Column(
            children: [
              // Header
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: cardColor,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 4,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    GestureDetector(
                      onTap: () => Navigator.pop(context),
                      child: Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                          color: primaryColor.withOpacity(0.1),
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          Icons.arrow_back,
                          color: primaryColor,
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Text(
                      '개인 설정',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: textColor,
                      ),
                    ),
                  ],
                ),
              ),

              // Content
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Description
                      Center(
                        child: Text(
                          '정확한 영양 분석을 위해 정보를 입력해주세요.',
                          style: TextStyle(
                            fontSize: 14,
                            color: textGray,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ),
                      const SizedBox(height: 32),

                      // 성별
                      Text(
                        '성별',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: textColor,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          Expanded(
                            child: _buildGenderButton(
                              0,
                              '남성',
                              primaryColor,
                              cardColor,
                              textColor,
                              textGray,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: _buildGenderButton(
                              1,
                              '여성',
                              primaryColor,
                              cardColor,
                              textColor,
                              textGray,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: _buildGenderButton(
                              2,
                              '미공개',
                              primaryColor,
                              cardColor,
                              textColor,
                              textGray,
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 32),

                      // 생년
                      Text(
                        '생년',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: textColor,
                        ),
                      ),
                      const SizedBox(height: 12),
                      _buildInputField(
                        placeholder: '예: 1990',
                        suffix: '년',
                        primaryColor: primaryColor,
                        cardColor: cardColor,
                        textColor: textColor,
                        textGray: textGray,
                      ),

                      const SizedBox(height: 24),

                      // 키와 몸무게
                      Row(
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  '키',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                    color: textColor,
                                  ),
                                ),
                                const SizedBox(height: 12),
                                _buildInputField(
                                  placeholder: '170',
                                  suffix: 'cm',
                                  primaryColor: primaryColor,
                                  cardColor: cardColor,
                                  textColor: textColor,
                                  textGray: textGray,
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  '몸무게',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                    color: textColor,
                                  ),
                                ),
                                const SizedBox(height: 12),
                                _buildInputField(
                                  placeholder: '65',
                                  suffix: 'kg',
                                  primaryColor: primaryColor,
                                  cardColor: cardColor,
                                  textColor: textColor,
                                  textGray: textGray,
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 32),

                      // 질환 선택
                      Text(
                        '가지고 있는 질환 (선택)',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: textColor,
                        ),
                      ),
                      const SizedBox(height: 12),
                      ...[
                        {'id': 'diabetes', 'label': '당뇨'},
                        {'id': 'hypertension', 'label': '고혈압'},
                        {'id': 'kidney', 'label': '신장 질환'},
                        {'id': 'allergy', 'label': '식품 알레르기'},
                        {'id': 'none', 'label': '없음'},
                      ].map((item) {
                        final isSelected = _conditions[item['id']]!;
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: GestureDetector(
                            onTap: () => _handleConditionToggle(item['id']!),
                            child: Container(
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                color: isSelected
                                    ? primaryColor.withOpacity(0.15)
                                    : cardColor,
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(
                                  color: isSelected
                                      ? primaryColor
                                      : Colors.transparent,
                                  width: isSelected ? 2 : 0,
                                ),
                              ),
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    item['label'] ?? '',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight:
                                          isSelected ? FontWeight.bold : FontWeight.normal,
                                      color: isSelected ? primaryColor : textColor,
                                    ),
                                  ),
                                  if (isSelected)
                                    Icon(
                                      Icons.check,
                                      color: primaryColor,
                                      size: 18,
                                    ),
                                ],
                              ),
                            ),
                          ),
                        );
                      }),

                      const SizedBox(height: 12),

                      // 기타 질환 입력
                      TextField(
                        enabled: !_conditions['none']!,
                        decoration: InputDecoration(
                          hintText: '기타 질환이나 알레르기 입력...',
                          filled: true,
                          fillColor: widget.isDarkMode
                              ? Colors.white.withOpacity(0.05)
                              : const Color(0xFFF9F9F7),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide: BorderSide.none,
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide: BorderSide.none,
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide:
                                BorderSide(color: primaryColor, width: 2),
                          ),
                          disabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: const EdgeInsets.all(16),
                          hintStyle: TextStyle(color: textGray),
                        ),
                        style: TextStyle(color: textColor),
                      ),

                      const SizedBox(height: 32),

                      // 저장하기 버튼
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: () => Navigator.pop(context),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: primaryColor,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 20),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                          child: const Text(
                            '저장하기',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
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
    );
  }

  Widget _buildGenderButton(
    int value,
    String label,
    Color primaryColor,
    Color cardColor,
    Color textColor,
    Color textGray,
  ) {
    final isSelected = _selectedGender == value;
    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedGender = value;
        });
      },
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: isSelected ? primaryColor.withOpacity(0.15) : cardColor,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? primaryColor : Colors.transparent,
            width: 2,
          ),
        ),
        child: Text(
          label,
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 14,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            color: isSelected ? primaryColor : textColor.withOpacity(0.7),
          ),
        ),
      ),
    );
  }

  Widget _buildInputField({
    required String placeholder,
    required String suffix,
    required Color primaryColor,
    required Color cardColor,
    required Color textColor,
    required Color textGray,
  }) {
    return TextField(
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        hintText: placeholder,
        suffixText: suffix,
        filled: true,
        fillColor: widget.isDarkMode
            ? Colors.white.withOpacity(0.05)
            : const Color(0xFFF9F9F7),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide.none,
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: primaryColor, width: 2),
        ),
        contentPadding: const EdgeInsets.all(16),
        hintStyle: TextStyle(color: textGray),
        suffixStyle: TextStyle(color: textGray),
      ),
      style: TextStyle(color: textColor),
    );
  }
}
