/// FraudX Analyst - History Screen
/// ==================================

import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/models.dart';
import '../config/api_config.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<HistoryItem> _history = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() { super.initState(); _loadHistory(); }

  Future<void> _loadHistory() async {
    setState(() { _isLoading = true; _error = null; });
    try {
      final history = await ApiService.getHistory(deviceId: ApiConfig.deviceId);
      if (mounted) setState(() { _history = history; _isLoading = false; });
    } catch (e) {
      if (mounted) setState(() { _error = e.toString(); _isLoading = false; });
    }
  }

  Future<void> _deleteItem(HistoryItem item) async {
    try {
      await ApiService.deleteHistoryItem(item.simulationId);
      if (mounted) setState(() => _history.removeWhere((h) => h.simulationId == item.simulationId));
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Record deleted'), backgroundColor: Color(0xFF2A9D8F)));
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e'), backgroundColor: const Color(0xFFEF4444)));
    }
  }

  Future<void> _clearAll() async {
    final confirmed = await showDialog<bool>(context: context, builder: (ctx) => AlertDialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      title: const Text('Clear History'), content: const Text('Delete all simulation history?'),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
        TextButton(onPressed: () => Navigator.pop(ctx, true), style: TextButton.styleFrom(foregroundColor: Colors.red), child: const Text('Delete All')),
      ],
    ));
    if (confirmed == true) {
      try {
        await ApiService.clearHistory(ApiConfig.deviceId);
        if (mounted) { setState(() => _history = []); ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('History cleared'), backgroundColor: Color(0xFF2A9D8F))); }
      } catch (e) { if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e'), backgroundColor: const Color(0xFFEF4444))); }
    }
  }

  void _showDetail(HistoryItem item) {
    showModalBottomSheet(
      context: context, isScrollControlled: true, backgroundColor: Colors.transparent,
      builder: (_) => _DetailSheet(item: item),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: SafeArea(child: Column(children: [
        // ── Header ──────────────────────────────────────────
        Padding(padding: const EdgeInsets.fromLTRB(20, 16, 20, 8), child: Row(children: [
          GestureDetector(
            onTap: () => Navigator.of(context).pop(),
            child: Container(width: 36, height: 36, decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 6)]),
              child: const Icon(Icons.arrow_back_ios_new, size: 16, color: Color(0xFF1A1A2E))),
          ),
          const SizedBox(width: 16),
          const Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Simulation History', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E))),
            Text('All past transaction simulations', style: TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
          ])),
          if (_history.isNotEmpty) GestureDetector(
            onTap: _clearAll,
            child: Container(width: 36, height: 36, decoration: BoxDecoration(color: const Color(0xFFFFEBEE), borderRadius: BorderRadius.circular(10)),
              child: const Icon(Icons.delete_outline, size: 18, color: Color(0xFFEF4444))),
          ),
        ])),

        // ── Content ─────────────────────────────────────────
        Expanded(
          child: _isLoading
              ? const Center(child: CircularProgressIndicator(color: Color(0xFF2A9D8F)))
              : _error != null
                  ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                      const Icon(Icons.cloud_off, size: 48, color: Color(0xFF9CA3AF)),
                      const SizedBox(height: 12),
                      Text('Unable to load history', style: TextStyle(color: Colors.grey.shade600)),
                      const SizedBox(height: 12),
                      ElevatedButton(onPressed: _loadHistory, style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF2A9D8F), foregroundColor: Colors.white), child: const Text('Retry')),
                    ]))
                  : _history.isEmpty
                      ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                          Icon(Icons.history, size: 56, color: Colors.grey.shade300),
                          const SizedBox(height: 12),
                          Text('No saved simulations yet.', style: TextStyle(fontSize: 15, color: Colors.grey.shade500)),
                          const SizedBox(height: 4),
                          Text('Run a simulation to see it here.', style: TextStyle(fontSize: 13, color: Colors.grey.shade400)),
                        ]))
                      : RefreshIndicator(
                          color: const Color(0xFF2A9D8F), onRefresh: _loadHistory,
                          child: ListView.builder(
                            padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
                            itemCount: _history.length,
                            itemBuilder: (ctx, i) {
                              final item = _history[i];
                              return Dismissible(
                                key: Key(item.simulationId),
                                direction: DismissDirection.endToStart,
                                onDismissed: (_) => _deleteItem(item),
                                background: Container(
                                  alignment: Alignment.centerRight,
                                  padding: const EdgeInsets.only(right: 20),
                                  margin: const EdgeInsets.only(bottom: 10),
                                  decoration: BoxDecoration(color: const Color(0xFFEF4444), borderRadius: BorderRadius.circular(16)),
                                  child: const Icon(Icons.delete, color: Colors.white),
                                ),
                                child: GestureDetector(
                                  onTap: () => _showDetail(item),
                                  child: _HistoryTile(item: item),
                                ),
                              );
                            },
                          ),
                        ),
        ),
      ])),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  final HistoryItem item;
  const _HistoryTile({required this.item});
  @override
  Widget build(BuildContext context) {
    final isFraud = item.isFraud;
    return Container(
      margin: const EdgeInsets.only(bottom: 10), padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
        border: Border(left: BorderSide(width: 4, color: isFraud ? const Color(0xFFEF4444) : const Color(0xFF4CAF50))),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 8, offset: const Offset(0, 2))]),
      child: Row(children: [
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Text('#${item.simulationId.substring(0, item.simulationId.length > 6 ? 6 : item.simulationId.length)}', style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
            const SizedBox(width: 8),
            Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
              decoration: BoxDecoration(color: isFraud ? const Color(0xFFFFEBEE) : const Color(0xFFE8F5E9), borderRadius: BorderRadius.circular(12)),
              child: Text(isFraud ? 'Fraud' : 'Safe', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: isFraud ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32)))),
          ]),
          const SizedBox(height: 6),
          Row(children: [
            Icon(Icons.access_time, size: 14, color: Colors.grey.shade400), const SizedBox(width: 4),
            Text(_formatDate(item.timestamp), style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
            const SizedBox(width: 12), Text('•', style: TextStyle(color: Colors.grey.shade400)), const SizedBox(width: 12),
            Text('${(item.riskScore * 100).toStringAsFixed(2)}%', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: isFraud ? const Color(0xFFEF4444) : Colors.grey.shade500)),
          ]),
        ])),
        Text('\$${item.transactionAmount.toStringAsFixed(2)}', style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
        const SizedBox(width: 6),
        Icon(Icons.chevron_right, size: 20, color: Colors.grey.shade400),
      ]),
    );
  }
  String _formatDate(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes} min ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${dt.day}/${dt.month}/${dt.year}';
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Detail Bottom Sheet
// ══════════════════════════════════════════════════════════════════════════════

class _DetailSheet extends StatelessWidget {
  final HistoryItem item;
  const _DetailSheet({required this.item});

  @override
  Widget build(BuildContext context) {
    final isFraud = item.isFraud;
    return DraggableScrollableSheet(
      initialChildSize: 0.7, minChildSize: 0.4, maxChildSize: 0.9,
      builder: (ctx, scrollController) {
        return Container(
          decoration: const BoxDecoration(color: Colors.white, borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
          child: SingleChildScrollView(
            controller: scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Center(child: Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.grey.shade300, borderRadius: BorderRadius.circular(2)))),
              const SizedBox(height: 20),

              // Title
              const Text('Simulation Detail', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 16),

              // Verdict
              Center(child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                decoration: BoxDecoration(color: isFraud ? const Color(0xFFFFEBEE) : const Color(0xFFE8F5E9), borderRadius: BorderRadius.circular(20)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(isFraud ? Icons.warning_amber : Icons.check_circle, color: isFraud ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32)),
                  const SizedBox(width: 8),
                  Text(isFraud ? 'FRAUD DETECTED' : 'SAFE TRANSACTION', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: isFraud ? const Color(0xFFD32F2F) : const Color(0xFF2E7D32))),
                ]),
              )),
              const SizedBox(height: 20),

              // Info rows
              _DetailRow(label: 'Simulation ID', value: item.simulationId),
              _DetailRow(label: 'Date / Time', value: '${item.timestamp.day}/${item.timestamp.month}/${item.timestamp.year}  ${item.timestamp.hour.toString().padLeft(2, '0')}:${item.timestamp.minute.toString().padLeft(2, '0')}'),
              if (item.modelUsed != null)
                _DetailRow(label: 'Model Used', value: item.modelUsed!),
              _DetailRow(label: 'Transaction Amount', value: '\$${item.transactionAmount.toStringAsFixed(2)}'),
              _DetailRow(label: 'Risk Score', value: '${(item.riskScore * 100).toStringAsFixed(2)}%'),
              _DetailRow(label: 'Prediction', value: item.predictionResult),
              const SizedBox(height: 16),

              // AI Explanation
              if (item.aiExplanation != null) ...[
                const Text('AI Explanation', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity, padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(color: const Color(0xFFF5F7FA), borderRadius: BorderRadius.circular(12)),
                  child: Text(item.aiExplanation!, style: const TextStyle(fontSize: 14, height: 1.5, color: Color(0xFF374151))),
                ),
              ],
              const SizedBox(height: 16),
            ]),
          ),
        );
      },
    );
  }
}

class _DetailRow extends StatelessWidget {
  final String label, value;
  const _DetailRow({required this.label, required this.value});
  @override
  Widget build(BuildContext context) {
    return Padding(padding: const EdgeInsets.only(bottom: 12), child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(label, style: const TextStyle(fontSize: 14, color: Color(0xFF6B7280))),
      Flexible(child: Text(value, textAlign: TextAlign.end, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF1A1A2E)))),
    ]));
  }
}
