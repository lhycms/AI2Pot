add_test([=[MtpQOpTest.apply]=]  /data/home/liuhanyu/hyliu/code/AI2Pot/source/build/bin/descriptor/mtp/test_mtpMRadialOp [==[--gtest_filter=MtpQOpTest.apply]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[MtpQOpTest.apply]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/AI2Pot/source/build/descriptor/mtp SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_mtpMRadialOp_TESTS MtpQOpTest.apply)
