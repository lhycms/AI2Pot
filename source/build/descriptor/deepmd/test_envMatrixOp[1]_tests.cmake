add_test([=[EnvMatrixOpTest.test_EnvMatrixFunction]=]  /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/bin/descriptor/deepmd/test_envMatrixOp [==[--gtest_filter=EnvMatrixOpTest.test_EnvMatrixFunction]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[EnvMatrixOpTest.test_EnvMatrixFunction]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/descriptor/deepmd SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_envMatrixOp_TESTS EnvMatrixOpTest.test_EnvMatrixFunction)
