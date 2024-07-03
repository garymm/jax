# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# To update Shardy to a new revision,
# a) update SHARDY_COMMIT to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/shardy/archive/<git hash>.tar.gz | sha256sum
#    and update XLA_SHA256 with the result.

SHARDY_COMMIT = "7afabee9bf7addaef719244fe0a605463738384d"
SHARDY_SHA256 = "0019dfc4b32d63c1392aa264aed2253c1e0c2fb09216f8e2cc269bbfb8bb49b5"

def repo():
    tf_http_archive(
        name = "shardy",
        sha256 = SHARDY_SHA256,
        strip_prefix = "shardy-{commit}".format(commit = SHARDY_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/shardy/archive/{commit}.tar.gz".format(commit = SHARDY_COMMIT)),
    )

    # For development, one often wants to make changes to the Shardy repository as well
    # as the JAX repository. You can override the pinned repository above with a
    # local checkout by either:
    # a) overriding the Shardy repository on the build.py command line by passing a flag
    #    like:
    #    python build/build.py --bazel_options=--override_repository=shardy=/path/to/shardy
    #    or
    # b) by commenting out the http_archive above and uncommenting the following:
    # local_repository(
    #    name = "shardy",
    #    path = "/path/to/shardy",
    # )
