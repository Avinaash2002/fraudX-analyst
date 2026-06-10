/// FraudX Analyst - Main Entry Point
/// ====================================

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/app_provider.dart';
import 'config/api_config.dart';
import 'screens/start_screen.dart';
import 'screens/home_screen.dart';
import 'screens/simulate_screen.dart';
import 'screens/train_screen.dart';
import 'screens/models_screen.dart';
import 'screens/history_screen.dart';
import 'screens/chat_screen.dart';
import 'widgets/animated_bot.dart';
import 'services/tutorial_service.dart';
import 'widgets/tutorial_overlay.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ApiConfig.init();
  runApp(const FraudXApp());
}

class FraudXApp extends StatelessWidget {
  const FraudXApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AppProvider()..loadModels()),
        ChangeNotifierProvider(create: (_) => TutorialService()),
      ],
      child: MaterialApp(
        title: 'FraudX Analyst',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF6366F1),
            brightness: Brightness.light,
          ),
          useMaterial3: true,
          appBarTheme: const AppBarTheme(
            centerTitle: true,
            elevation: 0,
          ),
        ),
        home: const StartScreen(),
        routes: {
          '/main': (context) => const MainScreen(),
          '/history': (context) => const HistoryScreen(),
        },
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Main Screen with Bottom Navigation + Floating Bot Icon
// ══════════════════════════════════════════════════════════════════════════════

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final List<Widget> _screens = const [
    HomeScreen(),
    SimulateScreen(),
    TrainScreen(),
    ModelsScreen(),
    ChatScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();
    final selectedIndex = provider.currentTabIndex;
    final isChatTab = selectedIndex == 4;
    final isKeyboardOpen = MediaQuery.of(context).viewInsets.bottom > 0;

    return Scaffold(
      body: Stack(
        children: [
          IndexedStack(
            index: selectedIndex,
            children: _screens,
          ),
          const TutorialOverlay(),
        ],
      ),

      // ── Floating Bot Icon (hidden on Chat tab) ─────────────
      floatingActionButton: Builder(
        builder: (ctx) {
          final tutorial = ctx.watch<TutorialService>();
          if (isChatTab || isKeyboardOpen) return const SizedBox.shrink();
          if (tutorial.isActive && tutorial.currentStep != 3) return const SizedBox.shrink();
          if (tutorial.isActive && tutorial.currentStep == 3) {
            return const _FlashingFab();
          }
          return BotFab(onTap: () => provider.switchTab(4));
        },
      ),

      bottomNavigationBar: Consumer<TutorialService>(
        builder: (context, tutorial, child) => IgnorePointer(
          ignoring: tutorial.isActive,
          child: Opacity(
            opacity: tutorial.isActive ? 0.4 : 1.0,
            child: NavigationBar(
              selectedIndex: selectedIndex,
              onDestinationSelected: (index) {
                provider.switchTab(index);
              },
              destinations: const [
                NavigationDestination(
                  icon: Icon(Icons.home_outlined),
                  selectedIcon: Icon(Icons.home),
                  label: 'Home',
                ),
                NavigationDestination(
                  icon: Icon(Icons.science_outlined),
                  selectedIcon: Icon(Icons.science),
                  label: 'Simulate',
                ),
                NavigationDestination(
                  icon: Icon(Icons.school_outlined),
                  selectedIcon: Icon(Icons.school),
                  label: 'Train',
                ),
                NavigationDestination(
                  icon: Icon(Icons.bar_chart_outlined),
                  selectedIcon: Icon(Icons.bar_chart),
                  label: 'Models',
                ),
                NavigationDestination(
                  icon: Icon(Icons.chat_outlined),
                  selectedIcon: Icon(Icons.chat),
                  label: 'Chat',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _FlashingFab extends StatefulWidget {
  const _FlashingFab();
  @override
  State<_FlashingFab> createState() => _FlashingFabState();
}

class _FlashingFabState extends State<_FlashingFab> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Container(
          width: 56,
          height: 56,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: const Color(0xFFFFE8D6),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFFFFE500).withOpacity(0.3 + _controller.value * 0.5),
                blurRadius: 12 + _controller.value * 16,
                spreadRadius: 2 + _controller.value * 6,
              ),
            ],
          ),
          child: const AnimatedBot(size: 36, isActive: true),
        );
      },
    );
  }
}