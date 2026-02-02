"""
ResultSaver：episodes.jsonl/task_summary.json 落盘
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time


@dataclass
class EpisodeReport:
    """单个 episode 的报告"""
    episode_id: int
    seed: Optional[int] = None
    success: bool = False
    metrics_read: Dict[str, Any] = None
    timing: Dict[str, float] = None
    task_name: str = ""
    episode_length: int = 0
    
    def __post_init__(self):
        if self.metrics_read is None:
            self.metrics_read = {}
        if self.timing is None:
            self.timing = {}


@dataclass
class TaskReport:
    """单个任务的报告"""
    task_name: str
    success_rate: float = 0.0
    avg_episode_length: float = 0.0
    metrics_agg: Dict[str, Any] = None
    failures: List[Dict[str, Any]] = None
    timing: Dict[str, float] = None
    total_episodes: int = 0
    
    def __post_init__(self):
        if self.metrics_agg is None:
            self.metrics_agg = {}
        if self.failures is None:
            self.failures = []
        if self.timing is None:
            self.timing = {}


class ResultSaver:
    """结果落盘（episode/task 两级）"""
    
    def __init__(self, output_dir: str = "./results"):
        """
        初始化 ResultSaver
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存 episode 数据，用于生成 task summary
        self.episodes_cache: Dict[str, List[EpisodeReport]] = {}
    
    def write_episode(self, episode_report: EpisodeReport) -> None:
        """
        写入单个 episode 结果到 episodes.jsonl
        
        Args:
            episode_report: episode 报告
        """
        try:
            # 构建文件路径
            episodes_file = self.output_dir / "episodes.jsonl"
            
            # 构建 episode 数据字典
            episode_data = {
                "episode_id": episode_report.episode_id,
                "seed": episode_report.seed,
                "success": episode_report.success,
                "metrics_read": episode_report.metrics_read,
                "timing": episode_report.timing,
                "task_name": episode_report.task_name,
                "episode_length": episode_report.episode_length,
                "timestamp": time.time()
            }
            
            # 追加写入到 episodes.jsonl
            with open(episodes_file, "a", encoding="utf-8") as f:
                json.dump(episode_data, f, ensure_ascii=False)
                f.write("\n")
            
            # 缓存 episode 数据
            if episode_report.task_name not in self.episodes_cache:
                self.episodes_cache[episode_report.task_name] = []
            self.episodes_cache[episode_report.task_name].append(episode_report)
            
            print(f"✅ Episode {episode_report.episode_id} 结果已保存到 {episodes_file}")
            
        except Exception as e:
            print(f"❌ 写入 episode 结果失败: {e}")
            import traceback
            traceback.print_exc()
    
    def write_task(self, task_name: Optional[str] = None, task_report: Optional[TaskReport] = None) -> None:
        """
        写入任务结果到 task_summary.json
        
        Args:
            task_name: 任务名称（如果需要从缓存生成报告）
            task_report: 任务报告（如果提供）
        """
        try:
            # 如果没有提供 task_report，但提供了 task_name，则从缓存生成
            if task_report is None and task_name is not None:
                task_report = self._generate_task_report(task_name)
            
            if task_report is None:
                print("⚠️ 缺少 task_report 或 task_name")
                return
            
            # 构建文件路径
            summary_file = self.output_dir / f"task_summary_{task_report.task_name}.json"
            
            # 构建任务数据字典
            task_data = {
                "task_name": task_report.task_name,
                "success_rate": task_report.success_rate,
                "avg_episode_length": task_report.avg_episode_length,
                "metrics_agg": task_report.metrics_agg,
                "failures": task_report.failures,
                "timing": task_report.timing,
                "total_episodes": task_report.total_episodes,
                "timestamp": time.time()
            }
            
            # 写入到 task_summary.json
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Task {task_report.task_name} 结果已保存到 {summary_file}")
            
        except Exception as e:
            print(f"❌ 写入任务结果失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_task_report(self, task_name: str) -> TaskReport:
        """
        从缓存的 episode 数据生成任务报告
        
        Args:
            task_name: 任务名称
        
        Returns:
            TaskReport: 生成的任务报告
        """
        episodes = self.episodes_cache.get(task_name, [])
        
        if not episodes:
            return TaskReport(task_name=task_name)
        
        # 计算成功率
        success_count = sum(1 for ep in episodes if ep.success)
        success_rate = success_count / len(episodes) if episodes else 0.0
        
        # 计算平均 episode 长度
        avg_episode_length = sum(ep.episode_length for ep in episodes) / len(episodes) if episodes else 0.0
        
        # 聚合指标
        metrics_agg = self._aggregate_metrics([ep.metrics_read for ep in episodes])
        
        # 收集失败案例
        failures = [
            {
                "episode_id": ep.episode_id,
                "metrics": ep.metrics_read,
                "timing": ep.timing
            }
            for ep in episodes if not ep.success
        ]
        
        # 聚合 timing
        timing = self._aggregate_timing([ep.timing for ep in episodes])
        
        return TaskReport(
            task_name=task_name,
            success_rate=success_rate,
            avg_episode_length=avg_episode_length,
            metrics_agg=metrics_agg,
            failures=failures,
            timing=timing,
            total_episodes=len(episodes)
        )
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合多个 episode 的指标
        
        Args:
            metrics_list: 指标列表
        
        Returns:
            Dict[str, Any]: 聚合后的指标
        """
        if not metrics_list:
            return {}
        
        agg_metrics = {}
        
        # 简单的平均值聚合
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in agg_metrics:
                        agg_metrics[key] = []
                    agg_metrics[key].append(value)
        
        # 计算平均值
        for key, values in agg_metrics.items():
            agg_metrics[key] = sum(values) / len(values)
        
        return agg_metrics
    
    def _aggregate_timing(self, timing_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        聚合多个 episode 的时间数据
        
        Args:
            timing_list: 时间数据列表
        
        Returns:
            Dict[str, float]: 聚合后的时间数据
        """
        if not timing_list:
            return {}
        
        agg_timing = {}
        
        # 简单的平均值聚合
        for timing in timing_list:
            for key, value in timing.items():
                if isinstance(value, (int, float)):
                    if key not in agg_timing:
                        agg_timing[key] = []
                    agg_timing[key].append(value)
        
        # 计算平均值
        for key, values in agg_timing.items():
            agg_timing[key] = sum(values) / len(values)
        
        return agg_timing
    
    def clear_cache(self, task_name: Optional[str] = None) -> None:
        """
        清除缓存的 episode 数据
        
        Args:
            task_name: 任务名称（如果为 None，则清除所有缓存）
        """
        if task_name:
            if task_name in self.episodes_cache:
                del self.episodes_cache[task_name]
        else:
            self.episodes_cache.clear()
        
        print(f"✅ 缓存已清除 {'for task ' + task_name if task_name else 'completely'}")
