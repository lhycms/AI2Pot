add_test([=[EnvMatrixTest.find_value_deriv]=]  /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/bin/descriptor/deepmd/test_envMatrix [==[--gtest_filter=EnvMatrixTest.find_value_deriv]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[EnvMatrixTest.find_value_deriv]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/descriptor/deepmd SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_envMatrix_TESTS EnvMatrixTest.find_value_deriv)
