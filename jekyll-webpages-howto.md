Information on how to build and update this Jekyll webpage 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
[the details given below are potentially Windows-specific, especially the installation part]


Install Ruby according to https://jekyllrb.com/docs/installation/windows/:

   1) Download and install a Ruby+Devkit version from RubyInstaller Downloads. Use default options for installation.
   2) Run the ridk install step on the last stage of the installation wizard. This is needed for installing gems with native extensions. You can find additional information regarding this in the RubyInstaller Documentation
   3) Open a new command prompt window from the start menu, so that changes to the PATH environment variable becomes effective. Install Jekyll and Bundler using "gem install jekyll bundler"
   4) Check if Jekyll has been installed properly: jekyll -v
   
   
In your repository checkout folder, run
   "rm Gemfile.lock" if this file exists already and then
   "bundle install"
   
Then look at the webpage locally by typing
   "bundle exec jekyll serve"
and finally copy the shown URL (http://127.0.0.something) to your browser. In some cases, you might need to use the provided local config file by typing `bundle exec jekyll serve --config _config_local.yml` instead.

All these steps are automatically running on the remote side when you push changes to the repository. So no need to do anything here but the above is handy to test your changes before to push.
