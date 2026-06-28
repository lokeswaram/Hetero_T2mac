@echo off
echo ===================================================
echo Cleaning up homogeneous and deprecated files...
echo ===================================================

:: Configs
del /f /q "src\config\algs\coma.yaml" 2>nul
del /f /q "src\config\algs\heterogeneous.yaml" 2>nul
del /f /q "src\config\algs\iql.yaml" 2>nul
del /f /q "src\config\algs\iql_beta.yaml" 2>nul
del /f /q "src\config\algs\qmix.yaml" 2>nul
del /f /q "src\config\algs\qmix_beta.yaml" 2>nul
del /f /q "src\config\algs\qtran.yaml" 2>nul
del /f /q "src\config\algs\test_write_new.yaml" 2>nul
del /f /q "src\config\algs\tmac_comm_rate.yaml" 2>nul
del /f /q "src\config\algs\tmac_full_comm.yaml" 2>nul
del /f /q "src\config\algs\tmac_p2p_comm.yaml" 2>nul
del /f /q "src\config\algs\tmac_vffac.yaml" 2>nul
del /f /q "src\config\algs\vdn.yaml" 2>nul
del /f /q "src\config\algs\vdn_beta.yaml" 2>nul
del /f /q "src\config\algs\vffac.yaml" 2>nul

:: Controllers
del /f /q "src\controllers\basic_controller.py" 2>nul
del /f /q "src\controllers\tmac_comm_rate_controller.py" 2>nul
del /f /q "src\controllers\tmac_full_comm_controller.py" 2>nul
del /f /q "src\controllers\tmac_vffac_controller.py" 2>nul
del /f /q "src\controllers\vffac_controller.py" 2>nul
del /f /q "src\controllers\tmac_p2p_comm_controller.py" 2>nul

:: Learners
del /f /q "src\learners\coma_learner.py" 2>nul
del /f /q "src\learners\q_learner.py" 2>nul
del /f /q "src\learners\qtran_learner.py" 2>nul
del /f /q "src\learners\tmac_comm_rate_learner.py" 2>nul
del /f /q "src\learners\tmac_full_comm_learner.py" 2>nul
del /f /q "src\learners\tmac_vffac_learner.py" 2>nul
del /f /q "src\learners\vffac_learner.py" 2>nul
del /f /q "src\learners\tmac_p2p_comm_learner.py" 2>nul

:: Agents & Modules
del /f /q "src\modules\agents\rnn_agent.py" 2>nul
del /f /q "src\modules\agents\rnn_msg_agent.py" 2>nul
del /f /q "src\modules\agents\tmac_comm_rate_rnn_msg_agent.py" 2>nul
del /f /q "src\modules\agents\tmac_full_comm_rnn_msg_agent.py" 2>nul
del /f /q "src\modules\agents\tmac_p2p_comm_rnn_msg_agent.py" 2>nul
del /f /q "src\modules\agents\tmac_rnn_agent.py" 2>nul
del /f /q "src\modules\agents\tmac_rnn_msg_agent.py" 2>nul
del /f /q "src\modules\agents\heterogeneous_agent.py" 2>nul
del /f /q "src\modules\agents\role_policy.py" 2>nul
del /f /q "src\modules\role_aware_attention.py" 2>nul
del /f /q "src\modules\trust_estimator.py" 2>nul

:: Critics
del /f /q "src\modules\critics\coma.py" 2>nul
del /f /q "src\modules\critics\role_aware_critic.py" 2>nul

:: Mixers
del /f /q "src\modules\mixers\qmix.py" 2>nul
del /f /q "src\modules\mixers\vdn.py" 2>nul
del /f /q "src\modules\mixers\qtran.py" 2>nul

:: Runners & Smac
del /f /q "src\runners\parallel_runner.py" 2>nul
del /f /q "src\smac\role_mapper.py" 2>nul
rmdir /s /q "src\smac" 2>nul

:: Root Obsolete Markdown & Files
del /f /q "BEGINNER_GUIDE.md" 2>nul
del /f /q "CHANGELOG_HETEROGENEOUS.md" 2>nul
del /f /q "DEPENDENCY_UPDATES.md" 2>nul
del /f /q "FINAL_ARCHITECTURE.md" 2>nul
del /f /q "FINAL_RUN_REPORT.md" 2>nul
del /f /q "MIGRATION_GUIDE.md" 2>nul
del /f /q "QUICK_START.md" 2>nul
del /f /q "README_HETEROGENEOUS.md" 2>nul
del /f /q "RUN_PROJECT_STEP_BY_STEP.md" 2>nul
del /f /q "TRAINING_COMMANDS.md" 2>nul
del /f /q "UNIFIED_DIFF_PATCHES.md" 2>nul
del /f /q "VERSION_COMPATIBILITY_REPORT.md" 2>nul
del /f /q "dependency_report.md" 2>nul
del /f /q "homogeneous_components.md" 2>nul
del /f /q "test.py" 2>nul

echo Clean complete!
pause
