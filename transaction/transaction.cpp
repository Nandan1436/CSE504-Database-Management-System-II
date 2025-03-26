#include<bits/stdc++.h>
using namespace std;

vector<vector<int>>graph;
vector<set<int>>graphNodes;
vector<vector<string>>matrix;
vector<pair<int,string>>timeline;
vector<string>seq;
vector<string>transID;

vector<char> color;
vector<int> parent;
int cycle_start, cycle_end;

void readInput(){
    ifstream file("input3.txt");
    string line;
    transID.push_back("");
    while(getline(file,line)){
        vector<string>row;
        stringstream ss(line);
        string value;
        getline(ss,value,',');
        transID.push_back(value);
        while(getline(ss,value,',')){
            row.push_back(value);
        }
        matrix.push_back(row);
    }
}

void createGraph(){
    graph.resize(matrix.size()+1);
    graphNodes.resize(matrix.size()+1);
    for(int j=0;j<matrix[0].size();j++){
        for(int i=0;i<matrix.size();i++){
            timeline.push_back({i+1,matrix[i][j]});
        }
    }
    for(int i=0;i<timeline.size();i++){
        if(timeline[i].second!="-"){
            for(int j=i+1;j<timeline.size();j++){
                if(timeline[i].first==timeline[j].first)continue;
                string temp=timeline[j].second;
                string temp2=timeline[i].second;
                temp.pop_back(),temp2.pop_back();
                temp = temp.substr(temp.find('(') + 1);
                temp2 = temp2.substr(temp2.find('(') + 1);
                if(temp==temp2 && ((timeline[i].second[0]=='R' && timeline[j].second[0]=='W') || (timeline[i].second[0]=='W'))){
                    if(graphNodes[timeline[i].first].find(timeline[j].first)==graphNodes[timeline[i].first].end()){
                        graph[timeline[i].first].push_back(timeline[j].first);
                        graphNodes[timeline[i].first].insert(timeline[j].first);
                    }

                }
            }
        }
    }
}

bool dfs(int v) {
    color[v] = 1;
    for (int u : graph[v]) {
        if (color[u] == 0) {
            parent[u] = v;
            if (dfs(u))
                return true;
        } else if (color[u] == 1) {
            cycle_end = v;
            cycle_start = u;
            return true;
        }
    }
    color[v] = 2;
    return false;
}

void findSequence(){
    vector<int>indegree(matrix.size()+1,0);
    vector<bool>visited(matrix.size()+1,false);
    queue<int>q;
    for (int i = 1; i < graph.size(); i++) {
        for (int neighbor : graph[i]) {
            indegree[neighbor]++;
        }
    }
    for(int i=1;i<indegree.size();i++){
        if(indegree[i]==0){
            q.push(i);
        }
    }
    while(!q.empty()){
        int node=q.front();
        q.pop();
        seq.push_back(transID[node]);
        for(int i=0;i<graph[node].size();i++){
            indegree[graph[node][i]]--;
            if(indegree[graph[node][i]]==0){
                q.push(graph[node][i]);
            }
        }
    }
    for(int i=0;i<seq.size();i++){
        cout<<seq[i];
        if(i!=seq.size()-1)cout<<"->";
    }
    cout<<"\n";
    
}

void checkCycle(){
    int n=graph.size();
    color.assign(n, 0);
    parent.assign(n, -1);
    cycle_start = -1;

    for (int v = 0; v < n; v++) {
        if (color[v] == 0 && dfs(v))
            break;
    }

    if (cycle_start == -1) {
        cout << "Conflict Serializable" << endl;
        findSequence();
    } else {
        vector<int> cycle;
        cycle.push_back(cycle_start);
        for (int v = cycle_end; v != cycle_start; v = parent[v])
            cycle.push_back(v);
        cycle.push_back(cycle_start);
        reverse(cycle.begin(), cycle.end());

        cout << "Not Conflict Serializable\n";
        cout << "Cycle: ";
        for (int v : cycle)
            cout << v << " ";
        cout << endl;
    }
}

void printGraph(){
    for(int i=1;i<graph.size();i++){
        cout<<i<<": ";
        for(int j=0;j<graph[i].size();j++){
            cout<<graph[i][j]<<" ";
        }
        cout<<"\n";
    }
}

int main(){
    readInput();
    createGraph();
    checkCycle();
    printGraph();
    
    return 0;
}