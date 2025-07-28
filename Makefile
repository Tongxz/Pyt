# 人体行为检测系统 Makefile
# Human Behavior Detection System Makefile

.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down docs

# 默认目标
help:
	@echo "人体行为检测系统开发工具"
	@echo "可用命令:"
	@echo "  install      - 安装生产依赖"
	@echo "  dev-install  - 安装开发依赖"
	@echo "  test         - 运行测试"
	@echo "  test-cov     - 运行测试并生成覆盖率报告"
	@echo "  lint         - 代码质量检查"
	@echo "  format       - 代码格式化"
	@echo "  clean        - 清理临时文件"
	@echo "  docker-build - 构建Docker镜像"
	@echo "  docker-up    - 启动Docker环境"
	@echo "  docker-down  - 停止Docker环境"
	@echo "  docs         - 生成文档"
	@echo "  setup-dev    - 一键设置开发环境"

# 安装依赖
install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock pytest-asyncio
	pip install black flake8 mypy isort bandit
	pip install pre-commit
	pre-commit install
	pip install -e .

# 测试
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# 代码质量
lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports
	@echo "Running bandit..."
	bandit -r src/ -f json -o bandit-report.json
	@echo "Running isort check..."
	isort --check-only --diff src/ tests/
	@echo "Running black check..."
	black --check --diff src/ tests/

# 代码格式化
format:
	@echo "Running isort..."
	isort src/ tests/
	@echo "Running black..."
	black src/ tests/

# 清理
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -f bandit-report.json

# Docker 操作
docker-build:
	docker build -t hbd:latest .

docker-build-dev:
	docker build -f Dockerfile.dev -t hbd:dev .

docker-up:
	docker-compose up -d

docker-up-dev:
	docker-compose --profile development up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec app bash

docker-dev-shell:
	docker-compose exec dev-tools bash

# 数据库操作
db-init:
	docker-compose exec postgres psql -U hbd_user -d hbd_db -f /docker-entrypoint-initdb.d/init_db.sql

db-shell:
	docker-compose exec postgres psql -U hbd_user -d hbd_db

db-backup:
	docker-compose exec postgres pg_dump -U hbd_user hbd_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

# 应用操作
run-api:
	python main.py --mode api

run-detection:
	python main.py --mode detection

run-demo:
	python main.py --mode demo

# 文档
docs:
	sphinx-build -b html docs/ docs/_build/

docs-serve:
	sphinx-autobuild docs/ docs/_build/

# 开发环境设置
setup-dev: dev-install
	@echo "创建必要的目录..."
	mkdir -p data/images data/videos data/models
	mkdir -p logs
	mkdir -p models
	@echo "设置Git hooks..."
	pre-commit install
	@echo "运行初始测试..."
	pytest tests/unit/ -v
	@echo "开发环境设置完成！"

# 生产部署
deploy-staging:
	@echo "部署到测试环境..."
	docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

deploy-prod:
	@echo "部署到生产环境..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 性能测试
perf-test:
	pytest tests/ -m "performance" -v

# 安全检查
security-check:
	bandit -r src/ -f json -o security-report.json
	safety check

# 依赖更新
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# 版本发布
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major

# 监控和日志
logs:
	tail -f logs/app.log

monitor:
	docker stats

# 备份和恢复
backup:
	@echo "创建备份..."
	mkdir -p backups
	tar -czf backups/backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ config/ models/

# 健康检查
health-check:
	curl -f http://localhost:8000/health || exit 1

# 完整的CI流程
ci: lint test-cov security-check
	@echo "CI检查完成！"

# 快速开始（新开发者）
quickstart: setup-dev docker-up
	@echo "等待服务启动..."
	sleep 10
	make health-check
	@echo "系统已就绪！访问 http://localhost:8000"