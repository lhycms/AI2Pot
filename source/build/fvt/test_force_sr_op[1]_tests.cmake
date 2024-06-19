add_test([=[ForceSrOpTest.forward_and_backward]=]  /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/bin/fvt/test_force_sr_op [==[--gtest_filter=ForceSrOpTest.forward_and_backward]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[ForceSrOpTest.forward_and_backward]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/fvt SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_force_sr_op_TESTS ForceSrOpTest.forward_and_backward)
