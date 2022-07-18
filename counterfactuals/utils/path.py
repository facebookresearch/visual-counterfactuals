# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class Path:
    @staticmethod
    def db_root_dir(db_name):
        if db_name == "CUB":
            return "/path/to/dataset"

        else:
            raise NotImplementedError

    @staticmethod
    def output_root_dir():
        return "/path/to/output/"
