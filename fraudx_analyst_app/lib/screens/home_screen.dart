/// FraudX Analyst - Home Screen
/// ===============================
/// Matches Lovable: gradient hero with decorative circles, stat cards
/// with correct colors (white/red/green/blue), staggered animations

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../services/api_service.dart';
import '../models/models.dart';
import '../config/api_config.dart';
import 'user_guide_screen.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/tutorial_service.dart';

// ── Lovable palette ─────────────────────────────────────────────────────────
const _kGreenTeal = Color(0xFF2A9D8F);
const _kSkyBlue = Color(0xFF38BDF8);
const _kLightGreen = Color(0xFF6BCB77);
const _kRed = Color(0xFFEF4444);
const _kGreen = Color(0xFF4CAF50);
const _kBlue = Color(0xFF3B82F6);
const _kDark = Color(0xFF1A1A2E);

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  late AnimationController _entrance;
  List<HistoryItem> _recentHistory = [];
  int _safeCount = 0;
  int _fraudCount = 0;
  double _totalProtected = 0;
  int _lastTabIndex = -1;
  bool _initialLoadDone = false;
  bool _isLoadingData = true;

  @override
  void initState() {
    super.initState();
    _entrance = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))..forward();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initialLoad().then((_) {
        if (mounted) _checkFirstTime();
      });
      context.read<AppProvider>().addListener(_onTabChanged);
    });
  }

  bool _isRefreshing = false;

  void _onTabChanged() {
    if (!mounted || !_initialLoadDone || _isRefreshing) return;
    final provider = context.read<AppProvider>();
    if (provider.currentTabIndex == 0 && _lastTabIndex != 0) {
      _isRefreshing = true;
      _refreshData().then((_) => _isRefreshing = false);
    }
    _lastTabIndex = provider.currentTabIndex;
  }

  // Called on first launch — retries for Render cold start
  Future<void> _initialLoad() async {
    if (mounted) setState(() => _isLoadingData = true);

    final provider = context.read<AppProvider>();
    bool backendReady = false;
    int retryCount = 0;

    while (!backendReady && retryCount < 10) {
      try {
        await provider.loadModels();
        await _loadHistory();
        backendReady = true;
      } catch (e) {
        retryCount++;
        debugPrint('Backend still waking up... Retry: $retryCount');
        await Future.delayed(const Duration(seconds: 3));
      }
    }

     if (mounted) {
      setState(() => _isLoadingData = false);
      _initialLoadDone = true;
    }

    if (!backendReady && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Backend server is taking too long to wake up'), backgroundColor: Colors.orange),
      );
    }
  }

  // Called on tab switch and pull-to-refresh — no retries needed
  Future<void> _refreshData() async {
    try {
      await _loadHistory();
      // Silently refresh models without triggering full rebuild
      final provider = context.read<AppProvider>();
      provider.loadModels();
    } catch (_) {}
  }

  Future<void> _loadHistory() async {
    try {
      final history = await ApiService.getHistory(deviceId: ApiConfig.deviceId);
      if (mounted) {
        setState(() {
          _recentHistory = history.take(3).toList();
          _safeCount = history.where((h) => !h.isFraud).length;
          _fraudCount = history.where((h) => h.isFraud).length;
          _totalProtected = history.fold(0.0, (sum, h) => sum + h.transactionAmount);
        });
      }
    } catch (_) {}
  }

  Future<void> _checkFirstTime() async {
    final tutorial = context.read<TutorialService>();
    final shouldShow = await tutorial.shouldShowTutorial();
    if (shouldShow && mounted) {
      tutorial.start();
    }
  }

  @override
  void dispose() {
    try { context.read<AppProvider>().removeListener(_onTabChanged); } catch (_) {}
    _entrance.dispose();
    super.dispose();
  }

  // Staggered slide-up animation helper
  Animation<Offset> _slideUp(double start, double end) {
    return Tween<Offset>(begin: const Offset(0, 0.15), end: Offset.zero).animate(
      CurvedAnimation(parent: _entrance, curve: Interval(start, end, curve: Curves.easeOut)));
  }
  Animation<double> _fade(double start, double end) {
    return Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _entrance, curve: Interval(start, end, curve: Curves.easeOut)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: SafeArea(
        child: Consumer<AppProvider>(
          builder: (context, provider, _) {
            if (_isLoadingData) {
              return Center(child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                const CircularProgressIndicator(color: _kGreenTeal),
                const SizedBox(height: 16),
                const Text('Connecting to server…', style: TextStyle(fontSize: 14, color: Color(0xFF9CA3AF))),
              ]));
            }
            return RefreshIndicator(
              color: _kGreenTeal,
              onRefresh: _refreshData,
              child: SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Header ──────────────────────────────────────
                  SlideTransition(
                    position: _slideUp(0.0, 0.3),
                    child: FadeTransition(
                      opacity: _fade(0.0, 0.3),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Row(children: [
                            Text('FraudX Analyst', style: TextStyle(fontSize: 21, fontWeight: FontWeight.w800, color: _kDark)),
                            SizedBox(width: 6),
                            Text('🛡️', style: TextStyle(fontSize: 18)),
                          ]),
                          GestureDetector(
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute(builder: (_) => const UserGuideScreen()),
                            ),
                            child: Container(
                              width: 44, height: 44,
                              decoration: BoxDecoration(
                                color: Colors.white, shape: BoxShape.circle,
                                boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 8)],
                                border: Border.all(color: Colors.grey.shade200.withOpacity(0.5)),
                              ),
                              child: const Center(child: Icon(Icons.info_outline, size: 22, color: _kDark)),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),

                  // ── Hero Card ───────────────────────────────────
                  SlideTransition(
                    position: _slideUp(0.1, 0.4),
                    child: FadeTransition(
                      opacity: _fade(0.1, 0.4),
                      child: _buildHeroCard(provider),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // ── Stats Grid ──────────────────────────────────
                  Row(children: [
                    // Safe Today — WHITE card
                    Expanded(child: SlideTransition(
                      position: _slideUp(0.2, 0.5),
                      child: FadeTransition(opacity: _fade(0.2, 0.5), child: _buildSafeTodayCard()),
                    )),
                    const SizedBox(width: 12),
                    // Fraud Blocked — RED card
                    Expanded(child: SlideTransition(
                      position: _slideUp(0.25, 0.55),
                      child: FadeTransition(opacity: _fade(0.25, 0.55), child: _buildFraudBlockedCard()),
                    )),
                  ]),
                  const SizedBox(height: 12),
                  Row(children: [
                    // Accuracy — GREEN card
                    Expanded(child: SlideTransition(
                      position: _slideUp(0.3, 0.6),
                      child: FadeTransition(opacity: _fade(0.3, 0.6), child: _buildAccuracyCard(provider)),
                    )),
                    const SizedBox(width: 12),
                    // Model Score — BLUE card
                    Expanded(child: SlideTransition(
                      position: _slideUp(0.35, 0.65),
                      child: FadeTransition(opacity: _fade(0.35, 0.65), child: _buildModelScoreCard(provider)),
                    )),
                  ]),
                  const SizedBox(height: 24),

                  // ── Recent Transactions ─────────────────────────
                  SlideTransition(
                    position: _slideUp(0.4, 0.7),
                    child: FadeTransition(
                      opacity: _fade(0.4, 0.7),
                      child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                        const Text('Recent Transactions', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: _kDark)),
                        GestureDetector(
                          onTap: () => Navigator.pushNamed(context, '/history'),
                          child: const Text('View All', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: _kGreenTeal)),
                        ),
                      ]),
                    ),
                  ),
                  const SizedBox(height: 12),
                if (_recentHistory.isEmpty)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                    child: Center(child: Text('No simulations yet. Run one to see results here.', style: TextStyle(fontSize: 13, color: Colors.grey.shade400))),
                  )
                else
                  ..._recentHistory.asMap().entries.map((entry) {
                    final i = entry.key;
                    final tx = entry.value;
                    final delay = 0.45 + (i * 0.05);
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: SlideTransition(
                        position: _slideUp(delay, delay + 0.25),
                        child: FadeTransition(
                          opacity: _fade(delay, delay + 0.25),
                          child: GestureDetector(
                            onTap: () {
                              showModalBottomSheet(
                                context: context,
                                backgroundColor: Colors.white,
                                shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
                                builder: (_) => Padding(
                                  padding: const EdgeInsets.all(24),
                                  child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
                                    Center(child: Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.grey.shade300, borderRadius: BorderRadius.circular(2)))),
                                    const SizedBox(height: 20),
                                    Row(children: [
                                      Icon(tx.isFraud ? Icons.warning_amber : Icons.check_circle, color: tx.isFraud ? _kRed : _kGreen, size: 28),
                                      const SizedBox(width: 12),
                                      Text(tx.isFraud ? 'FRAUD DETECTED' : 'SAFE TRANSACTION', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: tx.isFraud ? _kRed : _kGreen)),
                                    ]),
                                    const SizedBox(height: 20),
                                    _DetailRow(label: 'Simulation ID', value: tx.simulationId),
                                    _DetailRow(label: 'Amount', value: '\$${tx.transactionAmount.toStringAsFixed(2)}'),
                                    _DetailRow(label: 'Risk Score', value: '${(tx.riskScore * 100).toStringAsFixed(2)}%'),
                                    _DetailRow(label: 'Card Number', value: tx.cardNumber.toString()),
                                    _DetailRow(label: 'Time', value: _formatTimeAgo(tx.timestamp)),
                                    if (tx.aiExplanation != null) ...[
                                      const SizedBox(height: 16),
                                      const Text('AI Explanation', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: _kDark)),
                                      const SizedBox(height: 8),
                                      Container(
                                        padding: const EdgeInsets.all(14),
                                        decoration: BoxDecoration(color: const Color(0xFFF5F7FA), borderRadius: BorderRadius.circular(12)),
                                        child: Text(tx.aiExplanation.toString(), style: const TextStyle(fontSize: 13, height: 1.5, color: Color(0xFF374151))),
                                      ),
                                    ],
                                    const SizedBox(height: 16),
                                  ]),
                                ),
                              );
                            },
                            child: _TransactionTile(
                              id: '#${tx.simulationId.substring(0, tx.simulationId.length > 6 ? 6 : tx.simulationId.length)}',
                              status: tx.isFraud ? 'Fraud' : 'Safe',
                              amount: '\$${tx.transactionAmount.toStringAsFixed(2)}',
                              time: _formatTimeAgo(tx.timestamp),
                              riskPercent: '${(tx.riskScore * 100).toStringAsFixed(2)}%',
                              isFraud: tx.isFraud,
                            ),
                          ),
                        ),
                      ),
                    );
                  }),
                ],
              ),
              ),
            );
          },
        ),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  HERO CARD — gradient + decorative circles
  // ════════════════════════════════════════════════════════════════════════════

  Widget _buildHeroCard(AppProvider provider) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft, end: Alignment.bottomRight,
          colors: [_kGreenTeal, _kSkyBlue, _kLightGreen],
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: [BoxShadow(color: _kGreenTeal.withOpacity(0.3), blurRadius: 16, offset: const Offset(0, 6))],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: Stack(children: [
          // Decorative circles (matching Lovable)
          Positioned(right: -24, bottom: -24,
            child: Container(width: 120, height: 120, decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1), shape: BoxShape.circle))),
          Positioned(right: 40, top: -16,
            child: Container(width: 80, height: 80, decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.08), shape: BoxShape.circle))),

          // Content
          Padding(
            padding: const EdgeInsets.all(22),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(12)),
                  child: const Icon(Icons.shield_outlined, size: 20, color: Colors.white),
                ),
                const SizedBox(width: 10),
                Text('Protection Active', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.white.withOpacity(0.85))),
              ]),
              const SizedBox(height: 14),
              Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
                Text('\$${_totalProtected.toStringAsFixed(2)}', style: const TextStyle(fontSize: 34, fontWeight: FontWeight.w800, color: Colors.white, height: 1.1)),
                const Padding(padding: EdgeInsets.only(bottom: 4), child: Text('', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w400, color: Colors.white70))),
              ]),
              const SizedBox(height: 4),
              Text('Cumulative Sum of Simulated Transactions Analysed', style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.75))),
              const SizedBox(height: 16),
              GestureDetector(
                onTap: () => provider.switchTab(1),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(14)),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    const Icon(Icons.bolt, size: 16, color: Colors.white),
                    const SizedBox(width: 8),
                    const Text('Simulate Transaction', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Colors.white)),
                    const SizedBox(width: 6),
                    const Icon(Icons.chevron_right, size: 16, color: Colors.white),
                  ]),
                ),
              ),
            ]),
          ),
        ]),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  STAT CARDS — matching Lovable exactly
  // ════════════════════════════════════════════════════════════════════════════

  String _formatTimeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes} min ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${dt.day}/${dt.month}/${dt.year}';
  }

  // Safe Today — WHITE card, dark text
  Widget _buildSafeTodayCard() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 3))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: const Color(0xFFF1F5F9), borderRadius: BorderRadius.circular(10)),
            child: const Icon(Icons.shield_outlined, size: 18, color: _kDark),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(color: const Color(0xFFDCFCE7), borderRadius: BorderRadius.circular(8)),
            child: Text('${_safeCount + _fraudCount} total', style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Color(0xFF16A34A))),
          ),
        ]),
        const SizedBox(height: 14),
        Text('$_safeCount', style: const TextStyle(fontSize: 36, fontWeight: FontWeight.w800, color: _kDark)),
        const SizedBox(height: 2),
        Text('Safe Today', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.grey.shade600)),
      ]),
    );
  }

  // Fraud Blocked — SOLID RED
  Widget _buildFraudBlockedCard() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _kRed,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [BoxShadow(color: _kRed.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 3))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(10)),
            child: const Icon(Icons.gpp_maybe_outlined, size: 18, color: Colors.white),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(8)),
            child: Text(
              (_safeCount + _fraudCount) > 0 ? '${((_fraudCount / (_safeCount + _fraudCount)) * 100).toStringAsFixed(0)}%' : '0%',
              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.white),
            ),
          ),
        ]),
        const SizedBox(height: 14),
        Text('$_fraudCount', style: const TextStyle(fontSize: 36, fontWeight: FontWeight.w800, color: Colors.white)),
        const SizedBox(height: 2),
        Text('Fraud Blocked', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.white.withOpacity(0.85))),
      ]),
    );
  }

  // Accuracy — SOLID GREEN
  Widget _buildAccuracyCard(AppProvider provider) {
    final value = provider.bestModel != null
        ? '${(provider.bestModel!.accuracy * 100).toStringAsFixed(2)}%'
        : '—';
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _kGreen,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [BoxShadow(color: _kGreen.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 3))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(10)),
          child: const Icon(Icons.monitor_heart_outlined, size: 18, color: Colors.white),
        ),
        const SizedBox(height: 14),
        Text(value, style: const TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.white)),
        const SizedBox(height: 2),
        Text('Accuracy', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.white.withOpacity(0.85))),
      ]),
    );
  }

  // Model Score — SOLID BLUE
  Widget _buildModelScoreCard(AppProvider provider) {
    final value = provider.bestModel != null
        ? provider.bestModel!.aucRoc.toStringAsFixed(2)
        : '—';
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _kBlue,
        borderRadius: BorderRadius.circular(22),
        boxShadow: [BoxShadow(color: _kBlue.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 3))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(10)),
          child: const Icon(Icons.trending_up, size: 18, color: Colors.white),
        ),
        const SizedBox(height: 14),
        Text(value, style: const TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Colors.white)),
        const SizedBox(height: 2),
        Text('Model Score', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.white.withOpacity(0.85))),
      ]),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Transaction Tile
// ══════════════════════════════════════════════════════════════════════════════

class _TransactionTile extends StatelessWidget {
  final String id, status, amount, time, riskPercent;
  final bool isFraud;
  const _TransactionTile({required this.id, required this.status, required this.amount, required this.time, required this.riskPercent, required this.isFraud});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white, borderRadius: BorderRadius.circular(16),
        border: Border(left: BorderSide(width: 4, color: isFraud ? _kRed : _kGreen)),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 8, offset: const Offset(0, 2))],
      ),
      child: Row(children: [
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Text(id, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: _kDark)),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
              decoration: BoxDecoration(
                color: isFraud ? const Color(0xFFFFEBEE) : const Color(0xFFE8F5E9),
                borderRadius: BorderRadius.circular(12)),
              child: Text(status, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: isFraud ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32))),
            ),
          ]),
          const SizedBox(height: 6),
          Row(children: [
            Icon(Icons.access_time, size: 14, color: Colors.grey.shade400),
            const SizedBox(width: 4),
            Text(time, style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
            const SizedBox(width: 12),
            Text('•', style: TextStyle(color: Colors.grey.shade400)),
            const SizedBox(width: 12),
            Text(riskPercent, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: isFraud ? _kRed : Colors.grey.shade500)),
          ]),
        ])),
        Text(amount, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: _kDark)),
        const SizedBox(width: 8),
        Icon(Icons.chevron_right, size: 20, color: Colors.grey.shade400),
      ]),
    );
  }
}

class _DetailRow extends StatelessWidget {
  final String label, value;
  const _DetailRow({required this.label, required this.value});
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text(label, style: const TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
        Flexible(child: Text(value, textAlign: TextAlign.end, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF1A1A2E)))),
      ]),
    );
  }
}
