import 'package:flutter/material.dart';
import 'app_colors.dart';

class AppTheme {
  static const double borderRadius = 32.0;
  static const double cardBorderRadius = 24.0;
  static const double buttonBorderRadius = 20.0;

  // Sage Palette (Light)
  static ThemeData sageLightTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    scaffoldBackgroundColor: AppColors.sageBackground,
    colorScheme: const ColorScheme.light(
      primary: AppColors.sagePrimary,
      onPrimary: Colors.white,
      secondary: AppColors.sageSecondary,
      surface: AppColors.sageCardBg,
      onSurface: AppColors.sageTextDark,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.sagePrimary,
      foregroundColor: Colors.white,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.sageCardBg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(cardBorderRadius),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.sagePrimary,
        foregroundColor: Colors.white,
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(buttonBorderRadius),
        ),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.sageCardBg,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(buttonBorderRadius),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(buttonBorderRadius),
        borderSide: const BorderSide(color: AppColors.sagePrimary),
      ),
    ),
  );

  // Sage Palette (Dark)
  static ThemeData sageDarkTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.sageDarkBackground,
    colorScheme: const ColorScheme.dark(
      primary: AppColors.sageDarkPrimary,
      onPrimary: AppColors.sageDarkPrimaryText,
      secondary: AppColors.sageDarkSecondary,
      surface: AppColors.sageDarkCardBg,
      onSurface: AppColors.sageDarkText,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.sageDarkPrimary,
      foregroundColor: AppColors.sageDarkPrimaryText,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.sageDarkCardBg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(cardBorderRadius),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.sageDarkPrimary,
        foregroundColor: AppColors.sageDarkPrimaryText,
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(buttonBorderRadius),
        ),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.sageDarkCardBg,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(buttonBorderRadius),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(buttonBorderRadius),
        borderSide: const BorderSide(color: AppColors.sageDarkPrimaryText),
      ),
    ),
  );

  // Berry Palette (Light)
  static ThemeData berryLightTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    scaffoldBackgroundColor: AppColors.berryBackground,
    colorScheme: const ColorScheme.light(
      primary: AppColors.berryPrimary,
      onPrimary: Colors.white,
      secondary: AppColors.berrySecondary,
      surface: AppColors.berryCardBg,
      onSurface: AppColors.berryTextDark,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.berryPrimary,
      foregroundColor: Colors.white,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.berryCardBg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(cardBorderRadius),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.berryPrimary,
        foregroundColor: Colors.white,
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(buttonBorderRadius),
        ),
      ),
    ),
  );

  // Berry Palette (Dark)
  static ThemeData berryDarkTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.berryDarkBackground,
    colorScheme: const ColorScheme.dark(
      primary: AppColors.berryDarkPrimary,
      onPrimary: AppColors.berryDarkPrimaryText,
      secondary: AppColors.berryDarkSecondary,
      surface: AppColors.berryDarkCardBg,
      onSurface: AppColors.berryDarkText,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.berryDarkPrimary,
      foregroundColor: AppColors.berryDarkPrimaryText,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.berryDarkCardBg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(cardBorderRadius),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.berryDarkPrimary,
        foregroundColor: AppColors.berryDarkPrimaryText,
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(buttonBorderRadius),
        ),
      ),
    ),
  );

  // Midnight Palette (Light)
  static ThemeData midnightLightTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    scaffoldBackgroundColor: AppColors.midnightBackground,
    colorScheme: const ColorScheme.light(
      primary: AppColors.midnightPrimary,
      onPrimary: Colors.white,
      secondary: AppColors.midnightSecondary,
      surface: AppColors.midnightCardBg,
      onSurface: AppColors.midnightTextDark,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.midnightPrimary,
      foregroundColor: Colors.white,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.midnightCardBg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(cardBorderRadius),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.midnightPrimary,
        foregroundColor: Colors.white,
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(buttonBorderRadius),
        ),
      ),
    ),
  );

  // Midnight Palette (Dark)
  static ThemeData midnightDarkTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.midnightDarkBackground,
    colorScheme: const ColorScheme.dark(
      primary: AppColors.midnightDarkPrimary,
      onPrimary: AppColors.midnightDarkPrimaryText,
      secondary: AppColors.midnightDarkSecondary,
      surface: AppColors.midnightDarkCardBg,
      onSurface: AppColors.midnightDarkText,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.midnightDarkPrimary,
      foregroundColor: AppColors.midnightDarkPrimaryText,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: AppColors.midnightDarkCardBg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(cardBorderRadius),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.midnightDarkPrimary,
        foregroundColor: AppColors.midnightDarkPrimaryText,
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(buttonBorderRadius),
        ),
      ),
    ),
  );

  // Get theme based on palette and mode
  static ThemeData getTheme(String palette, bool isDark) {
    switch (palette) {
      case 'berry':
        return isDark ? berryDarkTheme : berryLightTheme;
      case 'midnight':
        return isDark ? midnightDarkTheme : midnightLightTheme;
      default: // sage
        return isDark ? sageDarkTheme : sageLightTheme;
    }
  }
}
