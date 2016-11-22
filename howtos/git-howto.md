How to update **from** the `master` or `development` branch
==============================================================================
Merging the public `development` or `master` branch into a _local, private_ 
branch is often done quite frequently and often generates undesired and 
unwarranted merge commits. Here is the solution. 

Short Version
---------------
Use:

```
git pull --rebase  # instead of git pull
```
or, more verbose:
```
git fetch development
git rebase development  # instead of git merge development
```

Long Version, Rationale
--------------------------
Pulling a public branch into a local private branch is reminiscent of `svn update`.
(The second pull case is a pull (request) into the publicly visible `development`
or `master` branch. This latter use-case is more like `svn commit`. In this case,
`pull` or `merge` are the right commands.) 

`git rebase branch-to-use-as-base` and `git pull --rebase` start from the HEAD of 
the `branch-to-use-as-base` and
add all differences of the given branch on top of it, as if these changes would have
been done and committed after the changes of the base branch. Thereby `rebase` not only 
changes the current branch (by using remote changes, like `merge` does), but also rewrites its 
history (which should rather not be done when the branch has become public). 
This is conceptually what `svn update` does: providing the most current base, on top of which a 
commit can be done. Putting it differently, in `svn` every  `merge` is in fact a `rebase`. 
This is almost always what we want if the branch is local and not accessible to any collaborators yet.
`--rebase`` can also be set as default option for `pull` in your local `git`.

See also: 
- https://git-scm.com/docs/git-rebase
- https://www.derekgourlay.com/blog/git-when-to-merge-vs-when-to-rebase/
- http://mislav.net/2013/02/merge-vs-rebase/
- https://coderwall.com/p/7aymfa/please-oh-please-use-git-pull-rebase
- https://www.atlassian.com/git/tutorials/merging-vs-rebasing/conceptual-overview
- http://stackoverflow.com/questions/2472254/when-should-i-use-git-pull-rebase

