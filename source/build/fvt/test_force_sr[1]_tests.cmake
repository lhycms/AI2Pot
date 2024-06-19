add_test([=[ForceSrTest.find_val_der]=]  /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/bin/fvt/test_force_sr [==[--gtest_filter=ForceSrTest.find_val_der]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ForceSrTest.find_val_der]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/fvt SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_force_sr_TESTS ForceSrTest.find_val_der)
