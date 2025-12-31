import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../theme/app_colors.dart';
import 'personal_settings_screen.dart';

class NutriSettingsScreen extends StatefulWidget {
  final String palette;
  final bool isDarkMode;

  const NutriSettingsScreen({
    super.key,
    required this.palette,
    required this.isDarkMode,
  });

  @override
  State<NutriSettingsScreen> createState() => _NutriSettingsScreenState();
}

class _NutriSettingsScreenState extends State<NutriSettingsScreen> {
  late String _selectedPalette;

  @override
  void initState() {
    super.initState();
    _selectedPalette = widget.palette;
  }

  @override
  Widget build(BuildContext context) {
    final theme = AppTheme.getTheme(_selectedPalette, widget.isDarkMode);
    final cardColor = theme.colorScheme.surface;
    final textColor = theme.colorScheme.onSurface;
    final textGray = theme.colorScheme.onSurface.withOpacity(0.6);

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
                      onTap: () {
                        Navigator.pop(context, {
                          'palette': _selectedPalette,
                          'isDarkMode': widget.isDarkMode,
                        });
                      },
                      child: Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                          color: theme.colorScheme.primary.withOpacity(0.1),
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          Icons.arrow_back,
                          color: theme.colorScheme.primary,
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Text(
                      '설정',
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
                      // Theme Section
                      Text(
                        '앱 테마 컬러',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: textGray,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 16),

                      // Palette options
                      GridView.count(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        crossAxisCount: 3,
                        mainAxisSpacing: 12,
                        crossAxisSpacing: 12,
                        children: [
                          _buildPaletteOption('sage', 'Sage', cardColor, textColor),
                          _buildPaletteOption('berry', 'Berry', cardColor, textColor),
                          _buildPaletteOption('midnight', 'Midnight', cardColor, textColor),
                        ],
                      ),

                      const SizedBox(height: 32),

                      // Account Section
                      Text(
                        '계정 및 알림',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: textGray,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 12),

                      ...['알림 설정', '계정 관리', '데이터 백업', '로그아웃'].map((item) {
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: GestureDetector(
                            onTap: () {
                              if (item == '계정 관리') {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => PersonalSettingsScreen(
                                      palette: _selectedPalette,
                                      isDarkMode: widget.isDarkMode,
                                    ),
                                  ),
                                );
                              }
                            },
                            child: Container(
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                color: cardColor,
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    item,
                                    style: TextStyle(
                                      fontSize: 16,
                                      color: textColor,
                                    ),
                                  ),
                                  Icon(
                                    Icons.chevron_right,
                                    color: textGray,
                                    size: 16,
                                  ),
                                ],
                              ),
                            ),
                          ),
                        );
                      }),
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

  Widget _buildPaletteOption(String key, String name, Color cardColor, Color textColor) {
    final isSelected = _selectedPalette == key;
    final theme = AppTheme.getTheme(key, widget.isDarkMode);

    Color primaryColor;
    switch (key) {
      case 'berry':
        primaryColor = widget.isDarkMode ? AppColors.berryDarkPrimary : AppColors.berryPrimary;
        break;
      case 'midnight':
        primaryColor = widget.isDarkMode ? AppColors.midnightDarkPrimary : AppColors.midnightPrimary;
        break;
      default:
        primaryColor = widget.isDarkMode ? AppColors.sageDarkPrimary : AppColors.sagePrimary;
    }

    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedPalette = key;
        });
      },
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: cardColor,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? theme.colorScheme.primary : Colors.transparent,
            width: 2,
          ),
        ),
        child: Column(
          children: [
            // Color preview circle
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: primaryColor,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),

            // Name
            Text(
              name,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: textColor,
              ),
            ),

            // Checkmark
            if (isSelected)
              const Padding(
                padding: EdgeInsets.only(top: 4),
                child: Icon(
                  Icons.check_circle,
                  size: 12,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
