# 参与开发

1. 对 planner、expert、geometry、controller、sampler 和 Local Goal 的修改必须具有通用性，禁止按任务 ID 或测试用例写特殊分支。
2. 修复缺陷时应增加回归测试，并报告精确的 PASS/FAIL 证据。
3. 禁止提交 API Key、Qwen 原始权重、HDF5 数据集、城市二进制资产、运行日志、缓存或生成的构建目录。
4. `DASHSCOPE_API_KEY` 或 `URBANFLY_QWEN_API_KEY` 只能通过进程环境传入。
5. 重新分发处理后的 Helsinki 资产时，必须保留 City of Helsinki 署名。
6. 不得把离线、仅负样本或 smoke 结果描述为闭环验收。

提交 Pull Request 前运行：

```bash
python -m pytest tests -q
cd frontend && npm test -- --run && npm run build
dotnet run --project desktop/UrbanFly.Desktop.Tests/UrbanFly.Desktop.Tests.csproj
```

已知的 CityGS Residence 测试依赖 Helsinki Release 未分发的旧资产。请将它单独记录为限制，不要静默跳过。
