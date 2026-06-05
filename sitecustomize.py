from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys


class _MarketStructure32Loader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        if hasattr(self.wrapped, "create_module"):
            return self.wrapped.create_module(spec)
        return None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        if getattr(module, "_case_explorer_attached", False):
            return
        original = module.render_structure32_section

        def render_with_explorer(st_module, plot_df, chart_renderer, *, chart_key=None):
            try:
                from structure32_case_explorer import render_structure32_case_explorer
                render_structure32_case_explorer(st_module, plot_df)
            except Exception as exc:
                try:
                    st_module.warning(f"32结构解释器加载失败：{exc}")
                except Exception:
                    pass
            return original(st_module, plot_df, chart_renderer, chart_key=chart_key)

        module.render_structure32_section = render_with_explorer
        module._case_explorer_attached = True


class _MarketStructure32Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "market_structure32":
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            if finder is importlib.machinery.PathFinder:
                spec = finder.find_spec(fullname, path, target)
            elif hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target)
            else:
                spec = None
            if spec is not None and spec.loader is not None:
                spec.loader = _MarketStructure32Loader(spec.loader)
                return spec
        return None


if not any(isinstance(finder, _MarketStructure32Finder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _MarketStructure32Finder())
